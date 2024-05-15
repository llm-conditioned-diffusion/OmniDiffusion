echo "Lauching Training"
python3 train_alignment.py \
            --llm_model_name_or_path baichuan-inc/Baichuan2-7B-Chat \
            --sd_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
            --en_train_data_path "./example_data/train_alignment_en_data.txt, ./example_data/train_alignment_en_data" \
            --ch_train_data_path "./example_data/train_alignment_cn_data.txt" \
            --wandb_api_key "set_your_wandb_api_key_here_to_track_training_procedure_on_your_wandb_account" \
            --test_length 40000 \
            --gpus 0 1 2 3 4 5 6 7 \
            --output_dir ./stage-1 \
            --model_save_name model_{epoch}-{step}_stage_1 \
            --real_batch_size 512 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --val_check_interval 0.1 \
            --learning_rate 2e-5 \
            --num_train_epochs 5 \
            --caption_loss cos-len \
            --strategy ddp