import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

def ScaledDotProduct(query, key, value, mask=None):
    dim_k = key.size(-1)
    scores = torch.bmm( query, key.transpose(1,2) ) / sqrt(dim_k)
    if mask is not None:
        scores = scores.mask_fill(mask==0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Produce tensors of shape [batch_size, seq_len, head_dim]
        # In practive head dim is a multiple of embed_dim 768/12 = 64
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state, mask=None):
        att_outputs = ScaledDotProduct(
                self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mask,
            )
        return att_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=2):
        super().__init__()
        embed_dim = hidden_size
        num_heads = num_attention_heads
        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
                [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
            )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state, mask=None):
        x = torch.cat( [h(hidden_state, mask) for h in self.heads], dim=-1 )
        x = self.output_linear(x)
        return x

