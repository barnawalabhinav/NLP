from attention import *
from position import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads)
        self.cross_attn = Attention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(X, tgt_mask)
        X = self.norm1(X + self.dropout(attn_output))
        attn_output = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.norm2(X + self.dropout(attn_output))
        ff_output = self.feed_forward(X)
        X = self.norm3(X + self.dropout(ff_output))
        return X