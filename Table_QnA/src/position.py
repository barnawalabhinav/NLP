import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoder, self).__init__()
        self.pos = nn.Parameter(torch.randn(max_seq_length, d_model))
        
    def forward(self, X):
        return X + self.pos[:, :X.size(1)]