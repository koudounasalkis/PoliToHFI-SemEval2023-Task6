from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch

class LJPEClassificationDataset(Dataset):
    def __init__(self, dataset_path, model_path='roberta-base', split="train", strategy="first"):
        self.split = split

        self.data = pd.read_csv(dataset_path)[['text', 'label', 'split']]
        self.data = self.data[self.data['split'] == self.split]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_len = self.tokenizer.model_max_length
        self.strategy = strategy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['text']
        label = item['label']
        if self.strategy == "first":            
            inputs = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=True,
                                padding='max_length',
                                verbose=False)
        elif self.strategy == "last":
            inputs = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=False,
                                padding='max_length',
                                verbose=False)
            new_inputs = {}
            new_inputs['input_ids'] = inputs['input_ids'][:, -self.max_len:]
            new_inputs['input_ids'][:, 0] = self.tokenizer.cls_token_id
            new_inputs['attention_mask'] = inputs['attention_mask'][:, -self.max_len:]
            if "token_type_ids" in inputs:
                new_inputs['token_type_ids'] = inputs['token_type_ids'][:, -self.max_len:]
            inputs = new_inputs
        
        inputs['input_ids'] = inputs['input_ids'].squeeze(0).long()  
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0).long() 
        if "token_type_ids" in inputs:
            inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0).long()
        inputs['labels'] = torch.tensor(label)
        return inputs