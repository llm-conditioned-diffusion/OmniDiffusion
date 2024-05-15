import torch

from text_encoder.mlm_encoder import MLMAdapter
from diffusers import UNet2DConditionModel


class UnetWithAdapter(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # preapre mlm adapter model        
        self.text_encoder_adapter = MLMAdapter(llm_hidden_size=4096)
        if args.mlm_ckpt:
            state_dict = torch.load(args.mlm_ckpt)
            self.text_encoder_adapter.load_state_dict(state_dict, strict=False)
            print("load adapter ckpt from ", args.mlm_ckpt)
            del state_dict
       
        # prepare unet 
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )
        if args.high_quality_tuning and args.unet_from_checkpoint is not None:
            ckpt = torch.load(args.unet_from_checkpoint)
            self.unet.load_state_dict(ckpt)
            print("load unet ckpt from ", args.unet_from_checkpoint)
            del ckpt
        
    def forward(self, noisy_latents, timesteps, last_hidden_list, seq_id_list, unet_added_conditions, weight_dtype=torch.float32):

        prompt_embed_list = []
        pooled_prompt_embed_list = []
        for last_hidden_state, seq_id in zip(last_hidden_list, seq_id_list):

            prompt_embed, pooled_prompt_embed = self.text_encoder_adapter(last_hidden_state, seq_id)
            prompt_embed_list.append(prompt_embed)
            pooled_prompt_embed_list.append(pooled_prompt_embed)

        prompt_embeds = torch.cat(prompt_embed_list, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embed_list, dim=0)       
        add_text_embeds = pooled_prompt_embeds
        add_text_embeds = add_text_embeds.to(self.unet.device, dtype=weight_dtype)
        prompt_hidden_states = prompt_embeds.to(self.unet.device, dtype=weight_dtype)

        unet_added_conditions["text_embeds"] = add_text_embeds

        model_pred = self.unet(
            noisy_latents, timesteps, prompt_hidden_states, added_cond_kwargs=unet_added_conditions
        ).sample

        return model_pred