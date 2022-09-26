'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
https://arxiv.org/abs/1706.03762
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import decoder
from encoder import encoder
from utils import PositionalEncoding 

class transformer(nn.Module):
    def __init__(self, d_shape:int):
        # Big Layers 
        self.PEnc = PositionalEncoding( d_shape ) 
        self.Encoder = encoder( d_shape )
        self.Decoder = decoder( d_shape )
        
        # Output Layers
        self.w = nn.Linear( d_shape, 1 )

    def forward( in_emb, out_emb ):
        # Encoding
        in_emb = Self.PEnc( in_emb )
        out_emb = Self.PEnc( out_emb )

        # Transformer Work
        e_output = self.Encoder( in_emb )
        d_output = self.Decoder( out_emb, e_output )

        # Output Layer
        return F.softmax( self.w( d_output ), dim=1 )
        

