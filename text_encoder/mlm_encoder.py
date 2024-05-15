import torch
import torch.nn as nn
import transformers

from typing import Optional, List
from dataclasses import dataclass, field
from lightning.pytorch import LightningModule
from transformers.training_args import TrainingArguments

from constants import *

class MLMAdapter(LightningModule):
    def __init__(self, llm_hidden_size=4096):
        super().__init__()

        sd_hidden_size = 2048
        hidden_size = 1280
        self.fc1 = nn.Sequential(
                nn.Linear(sd_hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ).to(TRAINABLE_PRECISION)

        self.max_token_length = 77
        self.learnable_prompt_length = 227
        self.t2i_decoder_prompt = torch.nn.Parameter(torch.randn((1, self.learnable_prompt_length, sd_hidden_size), dtype=TRAINABLE_PRECISION), requires_grad=True)
        self.llm_to_t2i_mapping = nn.Transformer(batch_first=True, norm_first=True, d_model = sd_hidden_size, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=sd_hidden_size*4, dropout=0.0, dtype=TRAINABLE_PRECISION)

        self.fc = nn.Sequential(
                    nn.Linear(llm_hidden_size, sd_hidden_size),
                    nn.GELU(),
                    nn.Linear(sd_hidden_size, sd_hidden_size),
                ).to(TRAINABLE_PRECISION)
        
    def forward(self, last_hidden_state, seq_id):
        t2i_input_embedding = self.fc(last_hidden_state[0:1, :seq_id])
        prompt_embeds = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt)
        pooled_prompt_embeds = torch.mean(prompt_embeds[:, 1:-1, :], dim=1) # Remove BOS and EOS, and calculate pooling features
        pooled_prompt_embeds = self.fc1(pooled_prompt_embeds)

        return prompt_embeds, pooled_prompt_embeds


class LLMPretrainedModel(LightningModule):
    def __init__(self, model_name_or_path, model_max_length) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['encoder_model_config'])
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model_max_length = model_max_length
        self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
            model_max_length=self.model_max_length, 
        )
    
        self.llm.requires_grad_(False)
        self.llm.to(PRECISION)
        self.llm.config.vocab_size = len(self.llm_tokenizer)
        self.stop_index = self.llm_tokenizer.eos_token_id

    def get_input_embeddings(self, input_ids):
        embed_tokens = self.llm.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        return inputs_embeds

    def forward(self, input_ids):

        bs_id, seq_id = (input_ids == self.stop_index).nonzero()[0]
        
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.device)
        input_embeds = self.get_input_embeddings(input_ids)
        outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                # labels=all_labels,
                output_hidden_states=True,
                past_key_values=None,
            )['hidden_states'][-1]

        last_hidden_state = outputs
        return last_hidden_state, seq_id


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="multimodal_encoder")
    model_name_or_path: Optional[str] = field(default="/cache/pretrained_model/Baichuan2-7B-Chat")
    model_save_name: str = field(default='model_{epoch}-{step}')
    use_sd_xl: bool = field(default=False)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default='results')
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size:int = field(default=2)
    per_device_eval_batch_size:int = field(default=2)
    real_batch_size:int = field(default=48)
    use_lora: bool = field(default=False)
    learning_rate:float = field(default=2e-5)
    warmup_ratio:float = field(default=0.03)
    warmup_steps:int = field(default=1000)
    adam_epsilon:float = field(default=1e-8)
    save_total_limit:int = field(default=1)
    gpus: List[int] = field(default_factory=[0,1,2,3,4,5,6,7])
    num_nodes: int = field(default=1)
    num_workers: int = field(default=16)
    strategy: str = field(default='ddp')
    val_check_interval: float = field(default=0.25)
    
    stage1_weight: Optional[str] = field(default=None)
    caption_loss: str = field(default='mse')
    visualize: bool = field(default=True)
    resume: Optional[str] = field(default=None)
    bilingual_alignment: bool = field(default=False)
    # en_ratio: bool = field(default=0.8)

