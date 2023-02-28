import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class h_transformer(nn.Module):

    def __init__(self, d_model: int = 768, nhead: int = 4, d_hid: int = 768,
                 nlayers: int = 1, dropout: float = 0.25):

        # PARAMETERS
        # d_model:  length of the tokens
        # nhead:    # of heads in the attention layer
        # d_hid:    dimension of the dense layer
        # n_layers: # of layers

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # TransformerEncoderLayer is a transformer layer made up of self attention + feed forward network
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # TransformerEncoder is a stack of N layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model


    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask = mask)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)