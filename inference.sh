#!/bin/bash
ROWS=1
COLS=1
STEPS=50

## 修改1
OUTPUT_DIR="./test"


MODEL_NAME="sdxl_base"

## 修改2

UNET="/zju_0038/diffusion_model/models/eccv_llm_diffusion/v2_s3_a=s2_d=v2/checkpoint-40/unet_ema/diffusion_pytorch_model.safetensors"
## 修改3
# 文件每行格式 id;;en_prompt;;negative_prompt;;ch_prompt 即以;;为分隔符,缺失的项直接跳过,保留分隔符即可
# 示例: 1141439418176131093;;sad indian man travelling around hong kong;;;;
INPUT_FILE="./benchmark_prompts.txt"

ADAPTER="/zju_0038/diffusion_model/models/eccv_llm_diffusion/v2_s3_a=s2_d=v2/checkpoint-40/text_encoder_adapter/pytorch_model.bin"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 python inference.py \
    --unet=${UNET} \
    --adapter_path=${ADAPTER} \
    --devices=0,1,2,3,4,5,6,7 \
    --resolution=1024 \
    --num_inference_steps=${STEPS} \
    --input_file=${INPUT_FILE} \
    --output_dir=${OUTPUT_DIR}