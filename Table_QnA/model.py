import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from embedding import Custom_Embedding


LOG_FILE = "train_log.txt"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):

    def __init__(self, max_seq_len: int = 256, d_model: int = 300, nhead: int = 2, d_hid: int = 128, nlayers: int = 2, dropout = 0.1, nclass=1):
        super(TransformerModel, self).__init__()
        self.nhead = nhead
        self.model_type = 'Transformer'
        self.embedding = Custom_Embedding(max_seq_len, d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=d_hid,
                                                dropout=dropout,
                                                batch_first=True,
                                                dtype=torch.float32,
                                                device=device,
                                                )
        self.layerNorm = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32, device=device)
        # self.batchNorm = nn.BatchNorm1d(max_seq_len, dtype=torch.float32, device=device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, norm=self.layerNorm, enable_nested_tensor=True)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1, device=device, dtype=torch.float32)
        # self.linear = nn.Linear(32, 1, device=device, dtype=torch.float32)
        self.down_scale = nn.Sigmoid()
        # self.down_scale = nn.SiLU()
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float32).unsqueeze(1).to(device))

        self.init_linear_weights()
        # self.transformer_encoder.apply(self.init_weights)

    def init_linear_weights(self) -> None:
        initrange = 0.01
        self.linear.bias.data.fill_(1.0)
        nn.init.xavier_normal_(self.linear.weight)
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def init_weights(self, m) -> None:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=0.01)
            # nn.init.kaiming_uniform_(m.weight)

    def forward(self, src, return_target = False):
        embed_src = self.embedding.forward(src, return_target)
        input = torch.stack([*embed_src['tokens'].values]).to(device)
        pad_mask = torch.stack([*embed_src['mask'].values]).to(device)
        attn_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2).to(device)
        attn_mask = attn_mask.repeat((self.nhead, 1, 1)).type(torch.bool).to(device)

        # target = torch.stack([*embed_src['target'].values]).to(device)

        # LOG_FILE.write(input)
        # LOG_FILE.write(input.shape)
        # LOG_FILE.write(attn_mask.shape)
        # LOG_FILE.write(target.shape)
        f = open(LOG_FILE, 'a')
        f.write(str((torch.max(input).item(), torch.min(input).item())) + '\n')
        output = self.transformer_encoder(input, attn_mask)
        output = self.layerNorm(output)
        f.write(str((torch.max(output).item(), torch.min(output).item())) + '\n')
        output = self.linear(output)
        output = output.squeeze(2)
        std = torch.std(output, dim=1).unsqueeze(1)
        mean = torch.mean(output, dim=1).unsqueeze(1)
        output = (output - mean) / std
        # output = self.batchNorm(output)
        f.write(str((torch.max(output).item(), torch.min(output).item())) + '\n')
        output = self.down_scale(output)
        f.write(str((torch.max(output).item(), torch.min(output).item())) + '\n')
        f.write("----------\n")
        f.close()
        return output, embed_src