'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
https://arxiv.org/abs/1706.03762
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.model.decoder import decoder
from transformer.model.encoder import encoder
from transformer.model.utils import PositionalEncoding 

class transformer(nn.Module):
    def __init__(self, d_shape:int, n_layers:int=2, word_window:int=512):
        super().__init__()
        # Big Layers 
        #self.PEnc = PositionalEncoding( d_shape ) 
        self.Encoder = encoder( d_shape, n_layers=n_layers)
        self.Decoder = decoder( d_shape, n_layers=n_layers)
        
        # Output Layers
        self.w = nn.Linear( d_shape * word_window, 1)

    def forward(self, in_emb, out_emb ):
        # Encoding

        #in_emb = Self.PEnc( in_emb )
        #out_emb = Self.PEnc( out_emb )

        # Transformer Work
        e_output = self.Encoder( in_emb )
        d_output = self.Decoder( out_emb, e_output )

        # Output Layer
        return F.softmax( self.w(torch.flatten(d_output, start_dim=1)), dim=1)

def get_model(shape:int):
    return transformer(shape)
        

