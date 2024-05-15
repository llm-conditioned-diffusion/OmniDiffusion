from torch.utils.data import Dataset
from .utils import *

class ValidationPromptDataset(Dataset):
    def __init__(
        self,
        filename,
    ):
        self.caption_cleaner = CleanCaption()
        self.prompt_list = []

        with open(filename, 'r') as ins:
            for it in ins:
                eles = it.strip(' \t\n').split(';;')
                if len(eles) <3:
                    continue
                en_prompt = eles[1]
                if not en_prompt:
                    continue
                self.prompt_list.append(eles[0:3])
                

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):
        
        retry_cnt = 0
        while True:
            if retry_cnt > 10 or index >= len(self.prompt_list):
                break
            if self.prompt_list[index]:
                pid, prompt, negative_prompt = self.prompt_list[index]
                return [pid, prompt, negative_prompt]
            else:
                index += 1
                retry_cnt += 1
