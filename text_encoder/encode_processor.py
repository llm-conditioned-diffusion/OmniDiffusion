import torch
import torch.distributed as dist


def is_main_process():
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        return True
    else:
        return False


def encode_prompt_with_mlm_adapter(mlm_encoder, tokenizer, adapter, prompts, device=None):
    """
    untrucated shape torch.Size([1, 21])
    text_encoder pooled shape torch.Size([1, 77, 768])
    text_encoder hiddend states shape torch.Size([1, 77, 768])
    bs_embed 1
    prompt_embeds shape torch.Size([1, 77, 768])
    untrucated shape torch.Size([1, 21])
    text_encoder pooled shape torch.Size([1, 1280])
    text_encoder hiddend states shape torch.Size([1, 77, 1280])
    bs_embed 1
    prompt_embeds shape torch.Size([1, 77, 1280])
    pooled_prompt_embeds shape torch.Size([1, 1280])
    prompt_embeds shape torch.Size([1, 77, 2048])
    """

    prompt_embed_list = []
    pooled_prompt_embed_list = []
    for _captions in prompts:
        with torch.no_grad():
            input_ids_ = tokenizer.encode(_captions, max_length=tokenizer.model_max_length)
            input_ids_ = [tokenizer.bos_token_id] + input_ids_ + [tokenizer.eos_token_id]
            input_ids_tensor = torch.LongTensor(input_ids_).unsqueeze(0)
            if device:
                input_ids_tensor = input_ids_tensor.to(device=device)

            last_hidden_state, seq_id = mlm_encoder(input_ids_tensor)
        prompt_embed, pooled_prompt_embed = adapter(last_hidden_state, seq_id)
        prompt_embed_list.append(prompt_embed)
        pooled_prompt_embed_list.append(pooled_prompt_embed)

    prompt_embeds = torch.cat(prompt_embed_list, dim=0)
    pooled_prompt_embeds = torch.cat(pooled_prompt_embed_list, dim=0)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt_without_adapter(mlm_encoder, tokenizer, prompts, device=None):
    """
    untrucated shape torch.Size([1, 21])
    text_encoder pooled shape torch.Size([1, 77, 768])
    text_encoder hiddend states shape torch.Size([1, 77, 768])
    bs_embed 1
    prompt_embeds shape torch.Size([1, 77, 768])
    untrucated shape torch.Size([1, 21])
    text_encoder pooled shape torch.Size([1, 1280])
    text_encoder hiddend states shape torch.Size([1, 77, 1280])
    bs_embed 1
    prompt_embeds shape torch.Size([1, 77, 1280])
    pooled_prompt_embeds shape torch.Size([1, 1280])
    prompt_embeds shape torch.Size([1, 77, 2048])
    """

    last_hidden_list = []
    seq_id_list = []
    for _captions in prompts:
        with torch.no_grad():
            input_ids_ = tokenizer.encode(_captions, max_length=tokenizer.model_max_length, truncation=True)
            input_ids_ = [tokenizer.bos_token_id] + input_ids_ + [tokenizer.eos_token_id]
            input_ids_tensor = torch.LongTensor(input_ids_).unsqueeze(0)
            if device:
                input_ids_tensor = input_ids_tensor.to(device=device)
            last_hidden_state, seq_id = mlm_encoder(input_ids_tensor)
            last_hidden_list.append(last_hidden_state)
            seq_id_list.append(seq_id)

    return last_hidden_list, seq_id_list

