from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch

class LJPEHierarchicalClassificationDataset(Dataset):
    def __init__(self, dataset_path, model_path='roberta-base', split="train", strategy="first", max_sentences=4):
        self.split = split
        self.data = pd.read_csv(dataset_path)
        self.data = self.data[self.data['split'] == self.split]
        self.data = self.data.groupby('doc_index')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_len = self.tokenizer.model_max_length
        self.strategy = strategy
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sents = self.data.get_group(idx)["sentence"].tolist()
        if len(sents) > self.max_sentences:
            if self.strategy == "first":
                sents = sents[:self.max_sentences]
            elif self.strategy == "last":
                sents = sents[-self.max_sentences:]
            else:
                raise ValueError("Strategy not supported")
        tokenized = self.tokenizer(sents, padding="max_length", truncation=True, return_tensors="pt")
        if len(sents) < self.max_sentences:
            pad_amount = self.max_sentences - len(sents)
            tokenized["input_ids"] = torch.cat([tokenized["input_ids"], torch.zeros(pad_amount, self.max_len, dtype=torch.long)], dim=0)
            tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], torch.zeros(pad_amount, self.max_len, dtype=torch.long)], dim=0)
        label = self.data.get_group(idx)["label"].tolist()[0]
        return tokenized, label