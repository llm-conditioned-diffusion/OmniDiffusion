#!/bin/bash
## modification 1 output directory
OUTPUT_DIR="./test"

## modification 2 model weight
# Demon
UNET="/path/to/unet/model"
ADAPTER="/path/to/adapter/module"

## modification 3 input prompts
# The format of each line in the file is as follows: 
# id;;en_prompt;;negative_prompt;;ch_prompt, where ;; serves as the delimiter. If any item is missing, simply skip it and retain the delimiter.
# Demonstration: 1141439418176131093;;sad indian man travelling around hong kong;;;;
INPUT_FILE="./example_data/prompts.txt"

RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 python inference.py \
    --unet=${UNET} \
    --adapter_path=${ADAPTER} \
    --devices=0,1,2,3,4,5,6,7 \
    --resolution=1024 \
    --num_inference_steps=50 \
    --input_file=${INPUT_FILE} \
    --output_dir=${OUTPUT_DIR}
