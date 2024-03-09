import math
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model = 64, num_heads = 8):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        Z = torch.matmul(attn_probs, V)
        return Z
    
    def split_heads(self, X):
        batch_size, seq_length, d_model = X.size()
        return X.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, X):
        batch_size, _, seq_length, d_k = X.size()
        return X.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, X, mask=None):
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        Z = self.W_o(self.combine_heads(attn_output))
        return Z