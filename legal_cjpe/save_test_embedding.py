import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForSequenceClassification
from argparse import ArgumentParser
from torch.utils.data import Dataset

from save_embeddings import SentenceDataset

class SentenceDataset(Dataset):
    def __init__(self, dataset_path, model_path='roberta-base', strategy="last", max_sentences=256):
        
        self.data = pd.read_csv(dataset_path)
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.groupby('doc_ids')
        self.group_indexes = list(self.data.groups.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_len = self.tokenizer.model_max_length
        self.strategy = strategy
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.group_indexes)

    def __getitem__(self, idx):
        doc_id = self.group_indexes[idx]
        sents = self.data.get_group(doc_id)["sentence"].tolist()
        if len(sents) > self.max_sentences:
            if self.strategy == "first":
                sents = sents[:self.max_sentences]
            elif self.strategy == "last":
                sents = sents[-self.max_sentences:]
            else:
                raise ValueError("Strategy not supported")
        tokenized = self.tokenizer(sents, padding="max_length", truncation=True, return_tensors="pt")
        return tokenized, doc_id

if __name__ == "__main__":

    parser = ArgumentParser(description='Training script')
    parser.add_argument('--tokenizer_path', 
        help='HF model name', 
        default="roberta-base", 
        required=False, 
        type=str)
    parser.add_argument('--sentence_encoder_path', 
        help='HF model name', 
        default="roberta-base", 
        required=False, 
        type=str)
    parser.add_argument('--sentences', 
        help='File name of the dataset', 
        default="testData/predict/test_files_CJP_sentences.csv", 
        required=False,
        type=str)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path        # e.g., 'roberta-base'
    sentence_encoder_path = args.sentence_encoder_path        # e.g., 'roberta-base'
    sentences = args.sentences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SentenceDataset(sentences, tokenizer_path, strategy="last", max_sentences=256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_dataset = SentenceDataset(sentences, tokenizer_path, strategy="last", max_sentences=256)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    sentence_encoder = AutoModelForSequenceClassification.from_pretrained(sentence_encoder_path, local_files_only=True)
    sentence_encoder.eval()
    sentence_encoder.to(device)

    tp = "single" if "single" in sentence_encoder_path else "multi"

    embeddings = []
    doc_ids = []
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        with torch.no_grad():
            tokenized, doc_id = batch
            input_ids = tokenized["input_ids"].view(-1, 512).to(device)
            attention_mask = tokenized["attention_mask"].view(-1, 512).to(device)
            output = sentence_encoder(input_ids, attention_mask, output_hidden_states=True)
            embeddings.append(output.hidden_states[-1][:, 0, :].cpu().numpy())
            doc_ids.append(doc_id)

    embeddings = np.array(embeddings)
    root_path = os.path.dirname(sentences)
    np.save(os.path.join(root_path, tp+"_test_embeddings.npy"), embeddings)
    with open(os.path.join(root_path, tp+"_test_doc_ids.txt"), "w") as f:
        for doc_id in doc_ids:
            f.write(doc_id[0] + "\n")
