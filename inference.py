import os
import random 
import argparse
import datetime
import numpy as np
from collections import defaultdict

import torch
import safetensors
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler
)

from text_encoder import LLMPretrainedModel, MLMAdapter
from dataset import ValidationPromptDataset
from dataset.utils import *
from pipeline import SDXLwithMLMPipeline

SEED_ID = 1106
torch.manual_seed(SEED_ID)
torch.cuda.manual_seed(SEED_ID)
np.random.seed(SEED_ID)
random.seed(SEED_ID)

def dataloader_init_fn():
    np.random.seed(SEED_ID)
    pass

torch.backends.cudnn.deterministic=True

model_zoo = {
    "sd-v1-5": "/cache/pretrained_model/stable-diffusion-v1-5", #pretrained model 
    "sdxl_base": "/cache/pretrained_model/sdxl-1.0/stable-diffusion-xl-base-1.0",
    "sdxl_refiner": "/cache/pretrained_model/sdxl-1.0/stable-diffusion-xl-refiner-1.0/",
    "floyd_stage_1": "/cache/pretrained_model/IF-I-XL-v1.0",
    "floyd_stage_2": "/cache/pretrained_model/IF-II-L-v1.0",
}

encoder_zoo = {
    "mlm_ckpt" : '/zju_0038/diffusion_model/models/llm_qformer/model-epoch_1-step_89929-Q_pt-0-60546_coyo_ft_vcg.ckpt',
    "mlm_pretrained_ckpt" : "/zju_0038/diffusion_model/models/Baichuan2-7B-Chat"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="input prompt files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cache/results/0905",
        help="image save dir",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="0,1",
        help="devices to run",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default=None,
        help="unet path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="llm adapter path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of inference step",
    )
    parser.add_argument(
        "--gs",
        type=float,
        default=7.5,
        help="guidance scale",
    )
    args = parser.parse_args()
    
    return args


def init_model(devices, model_name=None, sampling_method="Euler", torch_dtype=torch.float16):
    unet = None
    pipe = None 

    if model_name not in model_zoo:
        model_name = model_name
    else:
        model_name = model_zoo[model_name]
    assert model_zoo["unet"] is not None    
    print("model name = ", model_name)

    unet = UNet2DConditionModel.from_pretrained(model_zoo["sdxl_base"] + "/unet")
    print("ckpt = ", model_zoo["unet"])
    # get and load unet ckpt
    if model_zoo["unet"].endswith('safetensors'):
        ckpt = safetensors.torch.load_file(model_zoo["unet"], device='cpu')
    else:
        ckpt = torch.load(model_zoo["unet"]) # load pt file 
    if 'module' in ckpt.keys():
        unet.load_state_dict(ckpt['module'])
    else:
        unet.load_state_dict(ckpt)

    unet = unet.to(torch_dtype)
    del ckpt
    torch.cuda.empty_cache()

    pipe = SDXLwithMLMPipeline.from_pretrained(model_name, unet=unet, torch_dtype=torch_dtype)
    print("pipe = ", pipe)
    if sampling_method == "DPM":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampling_method == "DPMkarras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif sampling_method == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampling_method == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplementedError
    
    pipe.safety_checker = None
    pipe.to(devices[0])
    
    torch.cuda.empty_cache()
    return pipe

def latent_to_image(pipe, latents):
    with torch.no_grad():

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)


        # apply watermark if available
        if pipe.watermark is not None:
            image = pipe.watermark.apply_watermark(image)

        image = pipe.image_processor.postprocess(image.detach(), output_type='pil')

        return image

def main():
    args = parse_args()
    pmi_rank = int(os.environ['RANK'])
    pmi_wolrd_size = int(os.environ['WORLD_SIZE'])
    gpus_per_machine = torch.cuda.device_count()
    world_size = pmi_wolrd_size * gpus_per_machine
    if world_size == 1:
        worker(0)
    else:
        mp.spawn(worker, nprocs=gpus_per_machine, args=(pmi_rank, gpus_per_machine, world_size, args,))

def worker(gpu, pmi_rank, gpus_per_machine, world_size, args):
    ## init dist
    world_rank = pmi_rank * gpus_per_machine + gpu
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(
        backend='nccl', world_size=world_size, rank=world_rank,
        timeout=datetime.timedelta(hours=5)
    )

    guidance_scale = args.gs
    devices_vec = '{}'.format(gpu)
    target_size = [args.resolution, args.resolution]

    ## init dataloader
    valDataset = ValidationPromptDataset(args.input_file)
    sampler = DistributedSampler(valDataset)
    valid_dataloader = torch.utils.data.DataLoader(
                            valDataset,
                            batch_size=1,
                            sampler=sampler,
                            worker_init_fn=dataloader_init_fn,
                            shuffle=False)

    
    torch_type = torch.float16
    if args.torch_dtype == "float16":
        torch_type = torch.float16
    devices = ["cuda:{}".format(devices_vec[0])] 
    if len(devices_vec) > 1:
        devices.append("cuda:{}".format(devices_vec[1]))

    model_zoo["unet"] = None 
    
    if args.unet is not None:
        model_zoo["unet"] = args.unet
    
    model_name = "sdxl_base" # under no circumstance do you need to modify this
    num_inference_steps = args.num_inference_steps
    sampling_method = "DPMkarras" # "DPMkarras" # "DDIM" # "Euler" # "DPM"
    img_save_path = args.output_dir  

    if not os.path.exists(img_save_path):
        os.system("mkdir -p "+img_save_path)
    

    pipe = init_model(devices=devices, model_name=model_name, sampling_method=sampling_method, torch_dtype=torch_type)

    
    # mlm pretrained model
    text_encoder = LLMPretrainedModel(model_name_or_path=encoder_zoo['mlm_pretrained_ckpt'], model_max_length=227).to(devices[0])
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(torch_type)
    text_encoder.eval()

    # mlm adapter model
    adapter_path = args.adapter_path
    if adapter_path.endswith('safetensors'):
        state_dict = safetensors.torch.load_file(adapter_path, device='cpu') # load safetensors file
    elif adapter_path.endswith('bin'):
        state_dict = torch.load(adapter_path, map_location='cpu') # load pt file
    
    if adapter_path.endswith('ckpt'):
        text_encoder_adapter = MLMAdapter.load_from_checkpoint(adapter_path, strict=False, map_location="cpu", llm_hidden_size=4096)
    else: # adapter_path.endswith('bin' or 'safetensors')
        assert adapter_path.endswith('bin') or adapter_path.endswith('safetensors')
        text_encoder_adapter = MLMAdapter(llm_hidden_size=4096)
        text_encoder_adapter.load_state_dict(state_dict)

    text_encoder_adapter = text_encoder_adapter.to(devices[0])
    text_encoder_adapter = text_encoder_adapter.to(torch_type)
    text_encoder_adapter.eval()
    torch.cuda.empty_cache()


    pipe.init_custom_mlm_encoder(text_encoder, text_encoder_adapter, torch_dtype=torch_type)

    for ith_prompt, batch in enumerate(valid_dataloader):
        prompt_id, prompt, negative_prompt = batch
        prompt = list(prompt)
        prompt_id = list(prompt_id)
        negative_prompt = list(negative_prompt)
        print(f'rank {dist.get_rank()} iter {ith_prompt} prompt: {prompt}', flush=True)
        
        prompt2imgs = defaultdict(list)
         
        with torch.no_grad():
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, output_type="latent", guidance_scale=guidance_scale).images
            cur_img = latent_to_image(pipe, image.clone().detach())
            for rids, p in enumerate(prompt):
                prompt2imgs[p].append(cur_img[rids])

        for pdx, (p,img_list) in enumerate(prompt2imgs.items()):
            print(f'draw {p} \n img_list len {len(img_list)}', flush=True)
            grid_stage_1 = img_list[0]
            grid_stage_1.save(os.path.join(img_save_path, "{}.png").format(prompt_id[pdx]))

    torch.cuda.empty_cache()
    return 


if __name__ == "__main__":
    main()

