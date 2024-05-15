#!/usr/bin/env python
# coding=utf-8

import os
import math
import argparse
import logging
import random
import numpy as np
import datasets
import traceback

from PIL import Image
from pathlib import Path
from packaging import version
from datetime import timedelta
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms

import accelerate
import deepspeed
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import PretrainedConfig, CLIPTextModel
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from dataset import collate_fn, T2IDataset
from text_encoder import  ModelArguments, TrainingArguments
from text_encoder.encode_processor import encode_prompt_with_mlm_adapter, encode_prompt_without_adapter


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")
print(f'accelerate version {accelerate.__version__}', flush=True)
print(f'deepspeed version {deepspeed.__version__}', flush=True)

logger = get_logger(__name__, log_level="INFO")

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--stage-2", action="store_true")
    parser.add_argument("--stage-3", action="store_true")
    stage_arguments = parser.parse_args()
    if stage_arguments.stage_2:
        from configs.stage2_config import train_config
    elif stage_arguments.stage_3:
        from configs.stage3_config import train_config
    else:
        raise NotImplementError
    args = train_config
    if stage_arguments.stage_2 and args.high_quality_tuning:
        "Turn "

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f'accelerator state {accelerator.state}', main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """

        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        print(f'deepspeed_plugin is {deepspeed_plugin}', flush=True)
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        else:
            raise ValueError(f"{model_class} is not supported.")
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # import correct text encoder classes
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )


        # mlm pretrained model
        from text_encoder.mlm_encoder import LLMPretrainedModel, MLMAdapter
        text_encoder = LLMPretrainedModel(model_name_or_path=args.mlm_pretrained_ckpt, model_max_length=512)

    # super model init
    from models.super_model_unet import UnetWithAdapter
    super_unet = UnetWithAdapter(args)
    torch.cuda.empty_cache()

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    tokenizer = text_encoder.llm_tokenizer

    for param in super_unet.parameters():
        param.requires_grad = True

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        if args.high_quality_tuning and args.unet_from_checkpoint is not None:
            ema_unet.load_state_dict(torch.load(args.unet_from_checkpoint))
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            super_unet.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                save_step = int(accelerator.step)//args.gradient_accumulation_steps
                if args.resume_from_checkpoint:
                    path = os.path.basename(args.resume_from_checkpoint)
                    start_step = int(path.split("-")[1])
                    save_step += start_step
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(f'{args.output_dir}/checkpoint-{save_step}/', "unet_ema"))
                
                accelerator.save_model(super_unet.text_encoder_adapter, f'{args.output_dir}/checkpoint-{save_step}/text_encoder_adapter')
                accelerator.save_model(super_unet.unet, f'{args.output_dir}/checkpoint-{save_step}/unet')

        def load_model_hook(models, input_dir):
            if args.use_ema:
                if os.path.exists(f'{input_dir}/unet_ema'):
                    load_model = EMAModel.from_pretrained(f'{input_dir}/unet_ema', UNet2DConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    load_model_state_dict = torch.load(f'{input_dir}/pytorch_model/mp_rank_00_model_states.pt', map_location='cpu')
                    ema_unet.load_state_dict(load_model_state_dict['module'])
                    del load_model_state_dict
                ema_unet.to(accelerator.device)
                torch.cuda.empty_cache()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        super_unet.unet.enable_gradient_checkpointing() # to be changed 1

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    from accelerate.utils import DummyOptim, DummyScheduler, set_seed
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )

    optimizer = optimizer_cls(
        super_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    position_local_path = args.output_dir + '/pos/'
    os.makedirs(position_local_path, exist_ok=True)

    # Prepare training dataset and dataloader
    train_dataset = T2IDataset(
                    json_file=args.labeling_file, 
                    img_path=args.img_path, 
                    caption=args.caption_key_name
                )
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size,
                    shuffle=False,
                    collate_fn=lambda examples: collate_fn(examples),
                    num_workers=args.dataloader_num_workers,
                )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(f'len of dataloader is {len(train_dataloader)}')
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        print(f'scheduler is {args.lr_list}', flush=True)
        if args.custom_scheduler:
            from utils.lr_adjust import LRAdjuster
            lr_scheduler = LRAdjuster(args.lr_list, optimizer)
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.gradient_accumulation_steps,
                num_training_steps=args.lr_running_steps * accelerator.gradient_accumulation_steps,
            )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, 
            warmup_num_steps=args.lr_warmup_steps * accelerator.gradient_accumulation_steps,
            total_num_steps=args.lr_running_steps * accelerator.gradient_accumulation_steps, 
        )

    # Prepare everything with our `accelerator`.
    if args.custom_scheduler:
        super_unet, optimizer, train_dataloader = accelerator.prepare(
            super_unet, optimizer, train_dataloader
        )
    else:
        super_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            super_unet, optimizer, train_dataloader, lr_scheduler
        )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("lr_list")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if accelerator.state.deepspeed_plugin is None:
                accelerator.load_state(os.path.join(args.output_dir, path))
            else:
                deepspeed_config = {"load_optimizer_states": False, "load_lr_scheduler_states": False, "load_module_only": True}
                accelerator.load_state(args.resume_from_checkpoint, **deepspeed_config) # 否则lr_scheduler无法修改
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            torch.cuda.empty_cache()

            print(f'resume_global_step {resume_global_step}')
            print(f'first_epoch {first_epoch}')
            print(f'num_update_steps_per_epoch {num_update_steps_per_epoch}')
            print(f'resume_step {resume_step}')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    def compute_embeddings(text_encoder, tokenizer, prompts, batch, target_size):
        prompt_embeds, pooled_prompt_embeds = encode_prompt_with_mlm_adapter(text_encoder, tokenizer, super_unet.text_encoder_adapter, prompts, device=accelerator.device)
        add_text_embeds = pooled_prompt_embeds
            
        with torch.no_grad():
            def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids
            
            add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_size"], batch["crops_coords_top_left"])]
            )
            

            add_text_embeds = add_text_embeds.to(accelerator.device, dtype=weight_dtype)
            prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs
    
    def compute_embeddings_only(text_encoder, tokenizer, prompts, batch, target_size):
        last_hidden_list, seq_id_list = encode_prompt_without_adapter(text_encoder, tokenizer, prompts, device=accelerator.device)

        def compute_time_ids(original_size, crops_coords_top_left):
                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                return add_time_ids
        
        add_time_ids = torch.cat(
                [compute_time_ids(s, c) for s, c in zip(batch["original_size"], batch["crops_coords_top_left"])]
        )

        unet_added_cond_kwargs = {"time_ids": add_time_ids}

        return last_hidden_list, seq_id_list, unet_added_cond_kwargs

    for epoch in range(first_epoch, args.num_train_epochs):
        super_unet.train()
        train_loss = 0.0
        batch_iter = iter(train_dataloader)
        for step in range(len(train_dataloader)):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            batch = next(batch_iter)
            with accelerator.accumulate(super_unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents.to(weight_dtype)
                latents = latents * vae.config.scaling_factor #TODO: FP32 training
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                target_size = args.resolution
                
                last_hidden_list, seq_id_list, unet_added_conditions = compute_embeddings_only(
                    text_encoder, tokenizer,
                    batch["text_prompt"], batch, (target_size, target_size)
                )

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                 
                # Predict the noise residual and compute loss
                model_pred = super_unet(noisy_latents, timesteps, last_hidden_list, seq_id_list, unet_added_conditions, weight_dtype=weight_dtype)
                
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(super_unet.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.custom_scheduler:
                    gg_step = step + epoch * len(train_dataloader) 
                    gg_step = gg_step // int(args.gradient_accumulation_steps)
                    lr_scheduler.step(gg_step)
                else:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(super_unet.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.state.deepspeed_plugin:
                        accelerator.save_state(save_path)
                    elif accelerator.is_main_process:
                        accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    torch.cuda.empty_cache()

            if step > args.gradient_accumulation_steps:
                if args.resume_from_checkpoint and step <= args.gradient_accumulation_steps + resume_global_step:      
                    logs = {"step_loss": loss.clone().detach().item(), "lr": -1}
                else:
                    logs = {"step_loss": loss.clone().detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            else:
                logs = {"step_loss": loss.clone().detach().item(), "lr": -1}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(super_unet.unet.parameters())
                    ema_unet.copy_to(super_unet.unet.parameters())

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(super_unet.unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        super_unet = accelerator.unwrap_model(super_unet)
        if args.use_ema:
            ema_unet.copy_to(super_unet.unet.parameters())

        accelerator.save_model(super_unet.unet, f'{args.output_dir}/final_ema_model')

    accelerator.end_training()


if __name__ == "__main__":
    main()


