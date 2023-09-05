import torch
from torch import nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from linear import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Pre layer Norm
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model)
        self.feed_forward = FeedForward(d_model)

    def forward(self, x, mask=None):
        # Apply Layer Norm copy in q,k,v 
        hidden_state = self.layer_norm_1(x)

        # Apply attention with skip
        x = x + self.attention(hidden_state, mask)

        # Apply Feed Forward
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class TransformerEncoder(nn.Module):                                            
    def __init__(self, d_model, n_layers):                                                 
        super().__init__()                                                      
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model) for _ in range(n_layers)])

    def forward(self, x, mask=None):                                                       
        #x = self.embeddings(x)                                                  
        for layer in self.layers:                                               
            x = layer(x, mask)                                                        
        return x

