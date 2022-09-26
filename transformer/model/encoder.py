'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class encode_layer(nn.Module):
    def __init__(self, d_shape:int, d_forward:int=1024, heads:int=3):
        super().__init__()
        # Block 1
        self.mh_1 = multi_head( d_shape, d_forward, heads )
        self.ln_1 = nn.LayerNorm( d_shape )

        # Block 2
        self.w1 = nn.Linear( d_shape, d_forward )                               
        self.ln_2 = nn.LayerNorm( d_forward )                                   

        # Block 3                                                               
        self.w2 = nn.Linear( d_forward, d_shape )                             
        self.ln_3 = nn.LayerNorm( d_shape )
       

    def forward(self, in_vect, out_vect):
        block_1 = self.ln_1(
                    self.mh_1( in_vect, in_vect, in_vect ) + out_vect
                  )  

        block_2 = self.ln_2( self.w1( block_1 ) )
        block_3 = self.ln_3( self.w2( block_2 ) + block_1 )
        return block_3


class encoder(nn.Module):
    def __init__(self, d_shape ):
        super().__init__()
        # Both input and output vectors are the same shape
        self.e_layers = nn.ModuleList(
            [ encode_layer(d_shape) for i in range(6)]
        )

    def forward(self, in_v, out_v):
        x = out_v
        for f in self.d_layers:
            x = f( in_v, x)
        return self.w(x).reshape(x.shape[0],1)



