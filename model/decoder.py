'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.model.utils import *


class decode_layer(nn.Module):
    def __init__(self, d_shape:int, d_forward:int=1024, heads:int=3):
        super().__init__()
        # Block 1
        self.mh_1 = multi_head( d_shape, d_forward, heads )
        self.ln_1 = nn.LayerNorm( d_shape )
       
        # Block 2
        self.mh_2 = multi_head( d_shape, d_forward, heads )
        self.ln_2 = nn.LayerNorm( d_shape )

        # Block 3
        self.w1 = nn.Linear( d_shape, d_forward )
        self.ln_3 = nn.LayerNorm( d_forward )
        
        # Block 4
        self.w2 = nn.Linear( d_forward, d_shape )
        self.ln_4 = nn.LayerNorm( d_shape )

    def forward(self, out_vect, enc_vect):
        block_1 = self.ln_1(
                    self.mh_1( out_vect, out_vect, out_vect ) + out_vect
                  )  

        block_2 = self.ln_2(
                    self.mh_2( enc_vect, enc_vect, block_1) + block_1 
                  )  

        block_3 = self.ln_3( self.w1( block_2 ) )
        block_4 = self.ln_4( self.w2( block_3 ) + block_2 )
        return block_4


class decoder(nn.Module):
    def __init__(self, d_shape, n_layers:int=6):
        super().__init__()
        # Both input and output vectors are the same shape
        self.d_layers = nn.ModuleList(
            [ decode_layer(d_shape) for i in range(n_layers)]
        )
        #self.w = nn.Linear(d_shape, d_shape)

    def forward(self, out_v, enc_v):
        x = out_v 
        for f in self.d_layers:
            x = f( out_v, enc_v)
        #return self.w(x) 
        return x



