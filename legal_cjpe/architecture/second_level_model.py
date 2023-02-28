import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification

from architecture.hierarchical_transformer import h_transformer

class SecondLevelModel(nn.Module):
    
    def __init__(self, d_model: int = 768, nhead: int = 4, d_hid: int = 768,
                 nlayers: int = 1, dropout: float = 0.25, mlp_layers: int = 3):
        super().__init__()
        self.h_transformer = h_transformer(d_model, nhead, d_hid, nlayers, dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.mlp_layers = mlp_layers
        
    def forward(self, embeddings, attention_masks):
        
        # compute hierarchical encoding
        hierarchical_encoding = self.h_transformer(embeddings, attention_masks)

        # average pooling
        output = hierarchical_encoding.mean(dim=1)

        # compute output
        for _ in range(self.mlp_layers-1):
            output = self.fc(output)
            output = self.relu(output)
        output = self.fc_out(output)
        output = self.sigmoid(output)
        return output

