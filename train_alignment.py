import os
import math
import pathlib
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import json

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader, random_split
import transformers
from transformers.training_args import TrainingArguments
from torchvision import transforms

from models.sd_condition import SD_Condition
from dataset import AlignmentDataset
from dataset import AligmentDataCollator


def default_gpus():
    return [0,1,2,3]

@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Chat")
    sd_model_name_or_path: Optional[str] = field(default="stabilityai/stable-diffusion-xl-base-1.0")
    model_save_name: str = field(default='model_{epoch}-{step}')


@dataclass
class DataArguments:
    en_train_data_path: str = field(
        default=None, metadata={"help": "Path to the english training data."}
    )
    ch_train_data_path: str = field(
        default=None, metadata={"help": "Path to the chinese training data."}
    )
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    training_length: int = field(default=None)
    test_length: int = field(default=10000)


@dataclass
class TrainingArguments:
    seed: int = field(default=42)
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default='results')
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size:int = field(default=2)
    per_device_eval_batch_size:int = field(default=2)
    real_batch_size:int = field(default=48)
    learning_rate:float = field(default=2e-5)
    warmup_steps:int = field(default=1000)
    adam_epsilon:float = field(default=1e-8)
    gpus: List[int] = field(default_factory=default_gpus)
    num_nodes: int = field(default=1)
    num_workers: int = field(default=16)
    strategy: str = field(default='ddp')
    val_check_interval: float = field(default=0.1)
    
    stage1_weight: Optional[str] = field(default=None)
    caption_loss: str = field(default='mse')
    resume: Optional[str] = field(default=None)
    wandb_project_name: str = field(default="LLM_SD")
    wandb_api_key: str = field(default=None)

def make_supervised_data_module(data_args, training_args, data_collator, tokenizer) -> Dict:
    """Make dataset and collator for multilingual textual aligment."""
    test_length = data_args.test_length
    en_data_paths = [item.strip() for item in data_args.en_train_data_path.split(',')]
    ch_data_paths = [item.strip() for item in data_args.ch_train_data_path.split(',')]
    en_datasets, ch_datasets = [], []
    for data_path in en_data_paths:
        en_dataset = AlignmentDataset(data_path=data_path, tokenizer=tokenizer, language='en')
        en_datasets.append(en_dataset)
    for data_path in ch_data_paths:
        ch_dataset = AlignmentDataset(data_path=data_path, tokenizer=tokenizer, language='ch')
        ch_datasets.append(ch_dataset)

    en_ch_probs = [0.5, 0.5]
    train_dataloaders, val_datasets = [], []

    if en_ch_probs[1] > 1e-3: # multilingual
        for dataset in en_datasets + ch_datasets:
            train_bs = int(training_args.per_device_train_batch_size * en_ch_probs[0] / len(en_dataset)) if dataset.language == 'en' \
                            else int(training_args.per_device_train_batch_size * en_ch_probs[1] / len(ch_datasets))
            if train_bs < 1: train_bs = 1
            dataset_test_len = int(test_length * en_ch_probs[0] / len(en_datasets)) if dataset.language == 'en' \
                                else int(test_length * en_ch_probs[1] / len(ch_datasets))
            train_dataset, eval_dataset = random_split(dataset, lengths=[len(dataset)-dataset_test_len, dataset_test_len], generator=torch.Generator().manual_seed(3407))
            val_datasets.append(eval_dataset)
            train_dataloader = DataLoader(train_dataset, 
                                        batch_size=train_bs,
                                        num_workers=training_args.num_workers,
                                        collate_fn=data_collator,
                                        prefetch_factor=4,
                                        pin_memory=False)
            train_dataloaders.append(train_dataloader)
    else: # eng only
        for dataset in en_datasets:
            train_bs = int(training_args.per_device_train_batch_size * en_ch_probs[0] / len(en_dataset))
            if train_bs < 1: train_bs = 1
            dataset_test_len = int(test_length * en_ch_probs[0] / len(en_datasets)) if dataset.language == 'en' \
                                else int(test_length * en_ch_probs[1] / len(ch_datasets))
            train_dataset, eval_dataset = random_split(dataset, lengths=[len(dataset)-dataset_test_len, dataset_test_len], generator=torch.Generator().manual_seed(3407))
            val_datasets.append(eval_dataset)
            train_dataloader = DataLoader(train_dataset, 
                                        batch_size=train_bs,
                                        num_workers=training_args.num_workers,
                                        collate_fn=data_collator,
                                        prefetch_factor=4,
                                        pin_memory=False)
            train_dataloaders.append(train_dataloader)

    combined_val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    val_dataloader = DataLoader(combined_val_dataset, 
                                batch_size=training_args.per_device_eval_batch_size,
                                num_workers=training_args.num_workers,
                                collate_fn=data_collator,
                                prefetch_factor=4,
                                pin_memory=True)

    print("Data Loading Finished")
    return train_dataloaders, val_dataloader


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    seed_everything(training_args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["WANDB_API_KEY"] = training_args.wandb_api_key
    
    batch_size = training_args.real_batch_size
    devices = training_args.gpus
    num_devices = len(devices) * training_args.num_nodes
    gradient_accumulation_steps = max(1, batch_size // (training_args.per_device_train_batch_size*num_devices))

    if training_args.stage1_weight: # load a pretrained ckpt and finetune on it
        assert training_args.stage1_weight is not None
        model = SD_Condition.load_from_checkpoint(training_args.stage1_weight, strict=False, map_location="cpu", model_args=model_args, **vars(training_args))
    else: # train from scratch
        model = SD_Condition(model_args, **vars(training_args))
        
    tokenizer = model.llm_tokenizer
    data_collator = AligmentDataCollator(tokenizer)
    train_dataloaders, val_dataloaders = make_supervised_data_module(data_args, training_args, data_collator, tokenizer)
    
    checkpoint_callback = ModelCheckpoint(
                dirpath=training_args.output_dir,
                filename=model_args.model_save_name,
                monitor="val_loss",
                save_top_k=-1,
            )
    
    wandb_logger = WandbLogger(save_dir=training_args.output_dir, project=training_args.wandb_project_name, offline=True, name=model_args.model_save_name)
    trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=training_args.num_train_epochs, 
                        accumulate_grad_batches=gradient_accumulation_steps,
                        accelerator="gpu", devices=devices, 
                        num_nodes=training_args.num_nodes,
                        strategy=training_args.strategy,
                        logger=wandb_logger, 
                        precision='bf16-mixed',
                        val_check_interval=training_args.val_check_interval,
                        num_sanity_val_steps=0,
                        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_dataloaders, val_dataloaders, ckpt_path=training_args.resume)