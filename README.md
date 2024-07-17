# OmniDiffusion
This repo contains pytorch implementation, model weights, training/inference codes of ECCV 2024 accepted paper, "An Empirical Study and Analysis of Text-to-Image Generation Using Large Language Model-Powered Textual Representation", check our [project page](https://llm-conditioned-diffusion.github.io/) for more details.

## Dependencies and Installation
We suggest to use Anaconda to create a virtual python environment and install the corresponding dependencies:
```bash
conda create -n llm-condition python=3.9
conda activate llm-condition
git clone https://github.com/llm-conditioned-diffusion/llm-conditioned-diffusion.git
cd llm-conditioned-diffusion
pip install -r requirements.txt
```
## Model Weights of OmniDiffusion
Model weights can be downloaded from [huggingface](https://huggingface.co/Fudan-FUXI/llm-conditioned-diffusion-v1.0), we provide model weights trained after stage 1, stage 1 + stage 2, and stage 1 + stage 2 + stage 3, arrebeviated as ``stage 1'', ``stage 2'', and ``stage 3 '', respectively.
## Inference
To use our models to generate images, you need to modify and run the provided 'inference.sh', details of necessary modification are included in the script.
Particularly, you need to prepare a .txt file that contains the prompts used as image generation guidance. we provide [a template file](./example_data/prompts.txt) as demonstration.

## Training
### Config Setting
#### stage 1 config
To run the training code for stage 1, you need to specify the Chinese and English text data utilized for training (`en_train_data_path` and `cn_train_data_path`) in script `train_stage1.sh`.
You can set one of them as empty if your training data is pure English or pure Chinese.
#### stage 2 & stage 3 config
To run the training code for stage 2 and stage 3, there are some hyper-parameters to modify:

`mlm_ckpt` : model weights of the alignment adapter trained in stage-1.

`mlm_pretrained_ckpt`: model weights of the LLM text encoders, in our experiments, we use the powerful bilingual LLM [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat).

`pretrained_model_name_or_path`: model weights of [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

`high_quality_tuning`: set this as `True` if this config is used for stage 3 training, otherwise `False`.

`labeling_file`: meta data, we provide an example of this in `./example_data`.
`img_path`: the path to image files, should be consistent with the corresponding keys in the labeling_file.
`caption_key_name`: which key in the labeling files contains the caption, default as 'short_caption'.
### Training Data Prepartion
#### Data Preapartion for Alignment Training(Stage 1)
To train the alignment adapter, you need to provide plain text data. You can supply data by providing one or multiple text files or folders containing several text files. Please ensure that your data is in plain text format, and Each line in the text file should be a separate data sample.

You can specify one or multiple paths to your data files or folders.If you are providing multiple paths, separate them with commas.

For example, in `./example_data`, there is an English text file, `train_alignment_en_data.txt`, a directory contain 2 English text files, `train_alignment_en_data`, and a Chinese text file, `train_alignment_cn_data.txt`.
if you want to use them for stage 1 training, your `en_train_data_path` and `cn_train_data_path` shoubld be specified as: 
```python
    --en_train_data_path "./example_data/train_alignment_en_data.txt, ./example_data/train_alignment_en_data" \
    --ch_train_data_path "./example_data/train_alignment_cn_data.txt" \
```
#### Data Preparation for Text-to-Image Training(Stage 2 and Stage 3)
The instruction of data preparation for stage 2 and stage 3 training will come soon.
### Stage 1 Training
```bash
cd llm-conditioned-diffusion
sh train_stage1.sh
```
### Stage 2 Training
```bash
cd llm-conditioned-diffusion
sh train_stage2.sh
```
### Stage 3 Training 
```bash
cd llm-conditioned-diffusion
sh train_stage3.sh
```

## Bibtex
```
@article{tan2024empirical,
  title={An Empirical Study and Analysis of Text-to-Image Generation Using Large Language Model-Powered Textual Representation},
  author={Tan, Zhiyu and Yang, Mengping and Qin, Luozheng and Yang, Hao and Qian, Ye and Zhou, Qiang and Zhang, Cheng and Li, Hao},
  journal={arXiv preprint arXiv:2405.12914},
  year={2024}
}
```
