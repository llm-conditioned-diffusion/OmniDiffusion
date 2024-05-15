from typing import Any, Dict, List
from lightning.pytorch import LightningModule
import transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, get_linear_schedule_with_warmup, StoppingCriteria

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from constants import *
from text_encoder.raw_encoder import get_hidden_states_sdxl_lp, get_input_ids_lp


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False    
    

class SD_Condition(LightningModule):
    def __init__(self, model_args, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['encoder_model_config'])
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.llm_model_name_or_path,
            trust_remote_code=True,
        )
        self.model_max_length = 227 # 75 * 3 + 2, the desired alignment length
        self.learnable_prompt_length = self.model_max_length
        self.max_token_length = 77 # default for sd text encoder
        self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.llm_model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
            model_max_length=self.model_max_length, 
        )

        self.llm.requires_grad_(False)
        self.llm.to(PRECISION)
        self.stop_index = self.llm_tokenizer.eos_token_id

        sd_model_name = model_args.sd_model_name_or_path
        sd_text_encoder0 = CLIPTextModel.from_pretrained(sd_model_name, subfolder='text_encoder')
        sd_tokenizer0 = CLIPTokenizer.from_pretrained(sd_model_name, subfolder='tokenizer')
        sd_text_encoder1 = CLIPTextModelWithProjection.from_pretrained(sd_model_name, subfolder='text_encoder_2')
        sd_tokenizer1 = CLIPTokenizer.from_pretrained(sd_model_name, subfolder='tokenizer_2')
        sd_text_encoder0.requires_grad_(False)
        sd_text_encoder1.requires_grad_(False)
        sd_text_encoder0.to(PRECISION)
        sd_text_encoder1.to(PRECISION)
        self.tokenizers = [sd_tokenizer0, sd_tokenizer1]
        self.text_encoders = [sd_text_encoder0, sd_text_encoder1]

        sd_hidden_size = sd_text_encoder0.config.hidden_size + sd_text_encoder1.config.hidden_size
    
        self.fc1 = nn.Sequential(
                nn.Linear(sd_hidden_size, sd_text_encoder1.config.hidden_size),
                nn.GELU(),
                nn.Linear(sd_text_encoder1.config.hidden_size, sd_text_encoder1.config.hidden_size),
            ).to(TRAINABLE_PRECISION)
        
        self.t2i_decoder_prompt = torch.nn.Parameter(torch.randn((1, self.learnable_prompt_length, sd_hidden_size), dtype=TRAINABLE_PRECISION))
        self.llm_to_t2i_mapping = nn.Transformer(
                                    batch_first=True, 
                                    norm_first=True, 
                                    d_model = sd_hidden_size, 
                                    num_encoder_layers=4, 
                                    num_decoder_layers=4, 
                                    dim_feedforward=sd_hidden_size*4, 
                                    dropout=0.0, 
                                    dtype=TRAINABLE_PRECISION
                                    )
        self.fc = nn.Sequential(
                    nn.Linear(self.llm.config.hidden_size, sd_hidden_size),
                    nn.GELU(),
                    nn.Linear(sd_hidden_size, sd_hidden_size),
                ).to(TRAINABLE_PRECISION)
    
    def training_step(self, batch):
        final_batch = {}
        for key in batch[0].keys():
            if type(batch[0][key]) == list:
                final_batch[key] = []
                for i in range(len(batch)):
                    final_batch[key] += batch[i][key]
            else:
                final_batch[key] = torch.cat(tuple([item[key] for item in batch])).to (self.device)
        batch = final_batch
        
        input_captions = None
        captions = batch['caption']
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        bs = len(captions)

        loss_dict = self(input_ids, attention_mask, labels, captions)
        loss = loss_dict['loss']
        log_dict = {f'train_{k}': v.item() for k, v in loss_dict.items()}

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx: int=0):
        for key in batch.keys():
            if type(batch[key]) == list:
                batch[key] = batch[key]
            else:
                batch[key] = batch[key].to(self.device)
        
        captions = batch['caption']
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        loss_dict = self(input_ids, attention_mask, labels, captions)
        log_dict = {f'val_{k}': v.item() for k, v in loss_dict.items()}
        bs = len(captions)

        self.log_dict(log_dict, batch_size=bs, logger=True, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        optimizer_grouped_parameters = [
            {
                "params": self.t2i_decoder_prompt,
            },
            {
                "params": [p for n, p in self.fc.named_parameters() if p.requires_grad],
            },
            {
                "params": [p for n, p in self.fc1.named_parameters() if p.requires_grad],
            },
            {
                "params": [p for n, p in self.llm_to_t2i_mapping.named_parameters() if p.requires_grad],
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        return [optimizer], [scheduler]
    
    def get_input_embeddings(self, input_ids):
        embed_tokens = self.llm.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)
        
        return inputs_embeds
    
    def input_wrap(self, input_ids, attention_mask, labels):
        text_embeds = self.get_input_embeddings(input_ids)
        
        return text_embeds, attention_mask, labels
    

    def forward(self, input_ids, attention_mask, labels, captions) -> Any:
        batch_size = len(input_ids) if isinstance(input_ids, List) else input_ids.shape[0]
        all_input_embeds, all_attention, all_labels = [], [], []
        for b in range(batch_size):
            if isinstance(input_ids, List):
                wrapped_text_embeds, wrapped_atts_text, wrapped_labels = self.input_wrap(input_ids[b], attention_mask[b], labels[b])
            else:
                wrapped_text_embeds, wrapped_atts_text, wrapped_labels = self.input_wrap(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1])
            all_input_embeds.append(wrapped_text_embeds)
            all_attention.append(wrapped_atts_text)
            all_labels.append(wrapped_labels)

        #add padding features for batch
        max_len = max([x.shape[1] for x in all_input_embeds])
        for i in range(len(all_input_embeds)):
            if all_input_embeds[i].shape[1] < max_len:
                pad_len = max_len - all_input_embeds[i].shape[1]
                pad_embeds = torch.zeros([all_input_embeds[i].shape[0], pad_len, all_input_embeds[i].shape[2]]).to(all_input_embeds[i].device)
                pad_atts = torch.zeros([all_attention[i].shape[0], pad_len]).to(all_attention[i].device)
                pad_labels = torch.ones([all_labels[i].shape[0], pad_len], dtype=torch.long).to(all_labels[i].device) * -100
                all_input_embeds[i] = torch.cat([all_input_embeds[i], pad_embeds], dim=1)
                all_attention[i] = torch.cat([all_attention[i], pad_atts], dim=1)
                all_labels[i] = torch.cat([all_labels[i], pad_labels], dim=1)
        
        all_input_embeds = torch.cat(all_input_embeds, dim=0)
        all_attention = torch.cat(all_attention, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        outputs = self.llm(
            inputs_embeds=all_input_embeds,
            attention_mask=all_attention,
            return_dict=True,
            labels=all_labels,
            output_hidden_states=True,
            past_key_values=None,
        )

        last_hidden_state = outputs['hidden_states'][-1]
        caption_feature = []
        pooled_caption_embeds = []
        
        caption_feature, pooled_caption_embeds = self.encode_caption_xl_long_prompt(captions)
        t2i_input_embedding = self.fc(last_hidden_state)
        mapping_features = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt.repeat(batch_size, 1, 1), src_key_padding_mask=all_attention.to(t2i_input_embedding.dtype))
        segmented_mapping_features = mapping_features[:, 1:-1, :].reshape(batch_size, -1, self.max_token_length -2, mapping_features.shape[-1]) # Remove BOS and EOS, tri-segmented for calculating segemented pooling features
        pred_pooled_caption_embeds = torch.mean(segmented_mapping_features, dim=-2)
        pred_pooled_caption_embeds = self.fc1(pred_pooled_caption_embeds)
        pred_pooled_caption_embeds = pred_pooled_caption_embeds.reshape(batch_size, 3840) # 3840 = 3 * 1280

        if self.hparams.caption_loss == 'mse':
            mf0, mf1 = torch.split(mapping_features, [768, 1280], dim=-1)
            cf0, cf1 = torch.split(caption_feature, [768, 1280], dim=-1)
            caption_loss0 = F.mse_loss(mf0, cf0)
            caption_loss1 = F.mse_loss(mf1, cf1)
            caption_loss_pool = F.mse_loss(pred_pooled_caption_embeds, pooled_caption_embeds)
            
            loss = 0.1 * caption_loss0 + 0.1 * caption_loss1 + 0.1 * caption_loss_pool
            return {'loss': loss, 'caption_loss0': caption_loss0, 'caption_loss1': caption_loss1, 'caption_loss_pool': caption_loss_pool}
            
        elif self.hparams.caption_loss == 'euc':
            mf0, mf1 = torch.split(mapping_features, [768, 1280], dim=-1)
            cf0, cf1 = torch.split(caption_feature, [768, 1280], dim=-1)
            def euc_distance(input1, input2):
                res = torch.pow(input1 - input2, 2).sum(dim=-1, keepdim=False)
                res = torch.pow(res, 0.5)
                return torch.mean(res)
                
            caption_loss0 = euc_distance(mf0, cf0)
            caption_loss1 = euc_distance(mf1, cf1)
            caption_loss_pool = euc_distance(pred_pooled_caption_embeds, pooled_caption_embeds)
            loss = 0.1 * caption_loss0 + 0.1 * caption_loss1 + 0.1 * caption_loss_pool
            return {'loss': loss, 'caption_loss0': caption_loss0, 'caption_loss1': caption_loss1, 'caption_loss_pool': caption_loss_pool}
        
        elif self.hparams.caption_loss == 'cos':
            mf0, mf1 = torch.split(mapping_features, [768, 1280], dim=-1)
            cf0, cf1 = torch.split(caption_feature, [768, 1280], dim=-1)   
            caption_loss0 = 1. - F.cosine_similarity(mf0, cf0, dim=-1).mean()  
            caption_loss1 = 1. - F.cosine_similarity(mf1, cf1, dim=-1).mean()
            caption_loss_pool = 1. - F.cosine_similarity(pred_pooled_caption_embeds, pooled_caption_embeds, dim=-1).mean()
            
            loss = caption_loss0 + caption_loss1 + caption_loss_pool
            return {'loss': loss, 'caption_loss0': caption_loss0, 'caption_loss1': caption_loss1, 'caption_loss_pool': caption_loss_pool}
            
        elif self.hparams.caption_loss == 'cos-len':
            mf0, mf1 = torch.split(mapping_features, [768, 1280], dim=-1)
            cf0, cf1 = torch.split(caption_feature, [768, 1280], dim=-1)
            
            def cos_loss(input1, input2):
                loss_ = 1. - F.cosine_similarity(input1, input2, dim=-1).mean()
                
                return loss_
            
            def len_loss(input1, input2):
                loss_ = F.mse_loss(torch.norm(input1, dim=-1), torch.norm(input2, dim=-1))
                return 0.00002 * loss_

            caption_loss0_cos = cos_loss(mf0, cf0)
            caption_loss0_len = len_loss(mf0, cf0)
            caption_loss1_cos = cos_loss(mf1, cf1)
            caption_loss1_len = len_loss(mf1, cf1)
            caption_loss_pool_cos = cos_loss(pred_pooled_caption_embeds, pooled_caption_embeds)
            caption_loss_pool_len = len_loss(pred_pooled_caption_embeds, pooled_caption_embeds)
            loss = caption_loss0_cos + caption_loss0_len + caption_loss1_cos + caption_loss1_len + caption_loss_pool_cos + caption_loss_pool_len
               
            return {'loss': loss, 'caption_loss0_cos': caption_loss0_cos, 'caption_loss0_len': caption_loss0_len, 'caption_loss1_cos': caption_loss1_cos, 'caption_loss1_len': caption_loss1_len, 'caption_loss_pool_cos': caption_loss_pool_cos, 'caption_loss_pool_len': caption_loss_pool_len}

    def encode_caption_xl_long_prompt(self, prompt):
        if len(self.tokenizers) != 2 or len(self.text_encoders) != 2:
            print(f'there must be 2 text tokenizers and encoders', flush=True)
        
        ids_group = []
        for prompt in prompt:
            untruncated_ids0 = get_input_ids_lp(prompt, tokenizer=self.tokenizers[0])
            untruncated_ids1 = get_input_ids_lp(prompt, tokenizer=self.tokenizers[1])
            ids_group.append((untruncated_ids0, untruncated_ids1))
        
        untruncated_ids0, untruncated_ids1 = zip(*ids_group)
        untruncated_ids0 = torch.stack(untruncated_ids0, dim=0).to(self.device)
        untruncated_ids1 = torch.stack(untruncated_ids1, dim=0).to(self.device)

        hidden_emb1, hidden_emb2, pool2 = get_hidden_states_sdxl_lp(input_ids1=untruncated_ids0,
                                                                input_ids2=untruncated_ids1,
                                                                tokenizer1=self.tokenizers[0], 
                                                                tokenizer2=self.tokenizers[1],
                                                                text_encoder1=self.text_encoders[0].to(self.device),
                                                                text_encoder2=self.text_encoders[1].to(self.device),
                                                                )
        prompt_embeds = torch.concat([hidden_emb1, hidden_emb2], dim=-1)
        pooled_prompt_embeds = pool2.view(prompt_embeds.shape[0], -1)

        return prompt_embeds, pooled_prompt_embeds

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        trainable_param_names = [n for n, p in self.named_parameters() if p.requires_grad]
        # remove untrainable params
        for k in list(checkpoint["state_dict"].keys()):
            if k not in trainable_param_names:
                del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        #use pretrained weights for unsaved params
        current_state_dict = self.state_dict()
        state_dict = checkpoint["state_dict"]
        current_state_dict.update(state_dict)
        checkpoint["state_dict"] = current_state_dict 
