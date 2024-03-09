from attention import *
from position import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, mask):
        attn_output = self.self_attn(X, mask)
        X = self.norm1(X + self.dropout(attn_output))
        ff_output = self.feed_forward(X)
        X = self.norm2(X + self.dropout(ff_output))
        return X