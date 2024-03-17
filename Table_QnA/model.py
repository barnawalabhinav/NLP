import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from embedding import Custom_Embedding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):

    def __init__(self, max_seq_len: int = 256, d_model: int = 300, nhead: int = 2, d_hid: int = 300, nlayers: int = 2, dropout = 0.5):
        super(TransformerModel, self).__init__()
        self.nhead = nhead
        self.model_type = 'Transformer'
        self.embedding = Custom_Embedding(max_seq_len, d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                # dim_feedforward=d_hid,
                                                # dropout=dropout,
                                                batch_first=True,
                                                dtype=torch.float32,
                                                device=device
                                                )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1, device=device, dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

        self.init_linear_weights()
        # self.transformer_encoder.apply(self.init_weights)

    def init_linear_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(0, initrange)

    def init_weights(self, m) -> None:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=0.01)
            # nn.init.kaiming_uniform_(m.weight)

    def forward(self, src, return_target = False):
        embed_src = self.embedding(src, return_target)
        input = torch.stack([*embed_src['tokens'].values]).to(device)
        pad_mask = torch.stack([*embed_src['mask'].values]).to(device)
        attn_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2).to(device)
        attn_mask = attn_mask.repeat((self.nhead, 1, 1)).type(torch.bool).to(device)
        target = torch.stack([*embed_src['target'].values]).to(device)
        # print(input)
        # print(input.shape)
        # print(attn_mask.shape)
        # print(target.shape)
        print(torch.max(input).item(), torch.min(input).item())
        output = self.transformer_encoder(input, attn_mask)
        print(torch.max(output).item(), torch.min(output).item())
        output = self.linear(output)
        output = output.squeeze(2)
        print(torch.max(output).item(), torch.min(output).item())
        # output = self.sigmoid(output)
        # print(torch.max(output).item(), torch.min(output).item())
        print("----------")
        return output, target