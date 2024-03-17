import os
import math
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from embedding import Custom_Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class TransformerModel(nn.Module):

    def __init__(self, max_seq_len: int = 256, d_model: int = 300, nhead: int = 2, d_hid: int = 300, nlayers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = Custom_Embedding(max_seq_len, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, return_target = False) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, d_model]``
        """
        embed_src = self.embedding(src, return_target)
        target = embed_src['mask']
        output = self.transformer_encoder(embed_src['tokens'], embed_src['mask'])
        output = self.linear(output)
        return output, target