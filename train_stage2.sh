#!/bin/bash
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=./configs/deepspeed_config.json

accelerate launch --config_file=./configs/accelerate_config.yaml \
           train_t2i.py --stage-2
