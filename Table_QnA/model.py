import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from embedding import Custom_Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class TransformerModel(nn.Module):

    def __init__(self, max_seq_len: int = 256, d_model: int = 300, nhead: int = 2, d_hid: int = 300, nlayers: int = 2, dropout = 0.5):
        super(TransformerModel, self).__init__()
        self.nhead = nhead
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, dtype=torch.float32)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = Custom_Embedding(max_seq_len, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, return_target = False):
        embed_src = self.embedding(src, return_target)
        input = torch.stack([*embed_src['tokens'].values])
        pad_mask = torch.stack([*embed_src['mask'].values])
        attn_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        attn_mask = attn_mask.repeat((self.nhead, 1, 1)).type(torch.bool)
        target = torch.stack([*embed_src['mask'].values])
        # print(input)
        # print(input.shape)
        # print(attn_mask.shape)
        # print(target.shape)
        output = self.transformer_encoder(input, attn_mask)
        output = self.linear(output)
        return output, target