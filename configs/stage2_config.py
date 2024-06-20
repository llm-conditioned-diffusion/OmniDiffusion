from easydict import EasyDict
import json
import os 

def load_zero_config(filename):
    content = json.load(open(filename, 'r')) if os.path.exists(filename) else None
    return content

config = EasyDict()

config.mlm_ckpt = '/path/to/stage-1/adapter/pytorch_model.bin'
config.mlm_pretrained_ckpt = "/path/to/Baichuan2-7B-Chat"
config.pretrained_model_name_or_path = "/path/to/sdxl-1.0/stable-diffusion-xl-base-1.0"

config.job_name = "/zju_0038/diffusion_model/datasets/infos/datav2_2_list.txt"
config.task_name = "train_sdxl_with_mlm_adapter_use_cn_text"
config.cache_dir = None
config.output_dir = "./output"
config.local_file = True
config.revision = None # Revision of pretrained model identifier from huggingface.co/models.
config.unet_from_checkpoint = "/path/to/sdxl-1.0/stable-diffusion-xl-base-1.0/unet/pytorch_model.bin" ### change 2
config.resume_from_checkpoint = None

config.dataset_name = None
config.dataset_config_name = None
config.train_data_dir = None
config.image_column = "image"
config.caption_column = "text"
config.max_train_samples = None
config.validation_prompts = None

config.seed = None
config.input_perturbation = 0.1 #0
config.resolution = 1024
config.center_crop = True
config.random_flip = True
config.dataloader_num_workers = 0
config.aesthetic_score_th = 5.5

config.max_token_length = 227
config.truncate_size = 8192
config.zero_config_file = "./configs/deepspeed_config.json"
deepspeed_param = load_zero_config(config.zero_config_file)
config.train_batch_size = deepspeed_param['train_micro_batch_size_per_gpu'] if deepspeed_param else 1
config.gradient_accumulation_steps = deepspeed_param['gradient_accumulation_steps'] if deepspeed_param else 4
# config.train_batch_size = 1
# config.gradient_accumulation_steps = 1
print(f'config.gradient_accumulation_steps {config.gradient_accumulation_steps}', flush=True)
print(f'config.train_batch_size {config.train_batch_size}', flush=True)
config.num_train_epochs = 100
config.max_train_steps = 10000 # 150000

config.learning_rate = 1e-07
config.scale_lr = False
config.lr_scheduler = "cosine" #"constant"
config.lr_warmup_steps = 100
config.lr_running_steps = 10001

config.custom_scheduler = False

config.lr_list = [
        {"lr": 1e-7, "end_iter": 0+0,    "method": "constant"},
        {"lr": 3e-5, "end_iter": 0+1000,  "method": "linear"},
        {"lr": 3e-5, "end_iter": 0+12000, "method": "constant"},
        {"lr": 1e-5, "end_iter": 0+17000, "method": "cosine"},
        {"lr": 1e-7, "end_iter": 0+24000, "method": "cosine"},
    ]

config.snr_gamma = None # default: None
config.allow_tf32 = False
config.use_ema = True
config.non_ema_revision = None

config.mixed_precision = None
config.use_8bit_adam = False
config.adam_beta1 = 0.9
config.adam_beta2 = 0.999
config.adam_weight_decay = 0.03
config.adam_epsilon = 1e-08
config.max_grad_norm = 1.0

config.push_to_hub = False
config.hub_token = None
config.prediction_type = None
config.hub_model_id = None
config.logging_dir = "logs"
config.report_to = "tensorboard"

config.noise_offset = 0
config.local_rank = 1
config.checkpointing_steps = 1000 #10000
config.checkpoints_total_limit = None
config.validation_epochs = 5

config.enable_xformers_memory_efficient_attention = True
config.gradient_checkpointing = False

config.tracker_project_name = "text2image-fine-tune"

config.high_quality_tuning = False
config.labeling_file = "/zju_0038/mengping/dataset/high-quality/mjv6_aes_top20w/v2_merged_avail_data.json"
config.img_path = "/zju_0038/mengping/dataset/high-quality/v2_data_img"
config.caption_key_name = 'short_caption' # which key in the labeling files contains the caption, default as 'short_caption'

train_config = config
print(config)

