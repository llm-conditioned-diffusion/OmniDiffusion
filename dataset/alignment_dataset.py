import re
import os
import json
import torch
import random
import numpy as np
from typing import Any, List
from torch.utils.data import Dataset

from .utils import *
from constants import *



class AlignmentDataset(Dataset):
    def __init__(self, data_path, tokenizer, language='en', user_tokens=[195], assistant_tokens=[196]) -> None:
        super().__init__()

        self.file_list = []
        assert language in ['en', 'ch']
        self.language = language
        if language == 'en':
            self.encoding = 'latin-1'
        elif language == 'ch':
            self.encoding = 'utf-8'

        if data_path.endswith('txt'):
            self.file_list += [data_path]
        else:
            self.file_list += [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.txt')]
        
        random.shuffle(self.file_list)

        print('Start to load training data!')
        self.data = []
        for file_path in self.file_list:
            with open(file_path, mode='r', encoding=self.encoding) as f:
                for line in f:
                    sample = line.strip() # drop \n
                    if len(sample) != 0: 
                        self.data.append(sample)

        # load_data:
        datasetName = data_path.split('/')[-1]
        print(f'Training data {datasetName} loaded successfully!')
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.caption_cleaner = CleanCaption()
    
    def __len__(self):
        return len(self.data)
    
    def pre_caption(self, caption):
        
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        return caption
    
    def preprocessing_Q(self, example):
        # input_ids = []
        # labels = []
        '''
        system_prompt + user_tokens + user_prompts + assistant_tokens + target
        '''
        src_caption, tgt_caption = None, None
        if self.language == 'en':
            tgt_caption = src_caption = self.caption_cleaner.evaluate(example.strip())
        elif self.language == 'ch':
            tgt_caption, src_caption = [item for item in example.strip().split('\t')]
            tgt_caption = self.caption_cleaner.evaluate(tgt_caption.strip())
            src_caption = self.caption_cleaner.evaluate(src_caption.strip())

        input_ids = self.tokenizer(src_caption, 
                                  padding='max_length',
                                  max_length=self.tokenizer.model_max_length - 2,
                                  truncation=True,
                                  add_special_tokens=False).input_ids

        input_ids = [self.bos_id] + input_ids +[self.pad_id] * (self.tokenizer.model_max_length -2 - len(input_ids)) + [self.eos_id]   
        labels = [self.ignore_index] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        labels = torch.LongTensor(labels).unsqueeze(0)
        attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'caption': tgt_caption,
            'task_name': 'alignment'
        }
    
    def __getitem__(self, index) -> Any:
        try:
            return self.preprocessing_Q(self.data[index])
        except:
            print('Corrupted Data')
            return self.preprocessing_Q(self.data[index]+1)
