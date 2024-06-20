#!/bin/bash
ROWS=1
COLS=1
STEPS=50

## modification 1 output directory
OUTPUT_DIR="./test"


MODEL_NAME="sdxl_base"

## modification 2 model weight

UNET="/path/to/unet/model"
ADAPTER="/path/to/adapter/module"

## modification 3 input prompts
# 文件每行格式 id;;en_prompt;;negative_prompt;;ch_prompt 即以;;为分隔符,缺失的项直接跳过,保留分隔符即可
# 示例: 1141439418176131093;;sad indian man travelling around hong kong;;;;
INPUT_FILE="./example_data/prompts.txt"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 python inference.py \
    --unet=${UNET} \
    --adapter_path=${ADAPTER} \
    --devices=0,1,2,3,4,5,6,7 \
    --resolution=1024 \
    --num_inference_steps=${STEPS} \
    --input_file=${INPUT_FILE} \
    --output_dir=${OUTPUT_DIR}