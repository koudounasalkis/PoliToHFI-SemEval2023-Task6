from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class LJPESecondLevelClassificationDataset(Dataset):
    def __init__(self, embeddings, labels, strategy="last", max_sentences=256):

        print("Start Loading..")
        self.embeddings = embeddings
        self.labels = labels
        self.embedding_size = torch.from_numpy(self.embeddings[0]).shape[1]
        self.strategy = strategy
        self.max_sentences = max_sentences
        

    def __len__(self):
        return len(self.embeddings)


    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        embedding = torch.from_numpy(embedding)
        attention_mask = torch.ones(embedding.shape[0], dtype=torch.float)

        if embedding.shape[0] > self.max_sentences:
            if self.strategy == "first":
                embedding = embedding[:self.max_sentences]
                attention_mask = attention_mask[:self.max_sentences]
            elif self.strategy == "last":
                embedding = embedding[-self.max_sentences:]
                attention_mask = attention_mask[-self.max_sentences:]
            else:
                raise ValueError("Strategy not supported")
        elif embedding.shape[0] < self.max_sentences:
            pad_amount = self.max_sentences - len(embedding)
            embedding = torch.cat([embedding, torch.zeros(pad_amount, self.embedding_size, dtype=torch.float)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_amount, dtype=torch.float)], dim=0)

        label = self.labels[idx]
        return embedding, attention_mask, label