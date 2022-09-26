'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class scaled_dot(nn.Module):
    def forward(self, Q, K, V):
        # Q,K,V are all matricies (512x512)
        if False:
            print("dot product")
            print(Q.shape, K.shape, V.shape)
            print(K.transpose(1,2).shape)

        attention = torch.matmul( 
                        F.softmax(
                            torch.matmul(Q, K.transpose(1,2)) / K.shape[2],
                            dim=0
                        ),
                        V
                    )
        return attention


class single_multi_head(nn.Module):
    def __init__(self, emd_shape ):
        super().__init__()
        # k,q,v are all vectors???
        self.q_linear = nn.Linear(emd_shape, emd_shape)
        self.k_linear = nn.Linear(emd_shape, emd_shape)
        self.v_linear = nn.Linear(emd_shape, emd_shape)
        self.s_dot = scaled_dot()

    def forward(self, q, k, v):
        attention = self.s_dot(
                        self.q_linear(q),
                        self.k_linear(k),
                        self.v_linear(v),
                    )
        return attention


class multi_head(nn.Module):
    def __init__(self, d_shape:int, d_forward:int=1024, heads:int=3):
        super().__init__()
        self.heads = nn.ModuleList(
            [ single_multi_head(d_shape) for i in range(heads)]       
        )
        #self.w1 = nn.Linear((heads*d_shape), d_forward)
        self.w1 = nn.Linear((heads*d_shape), d_shape)
        self.drop1 = nn.Dropout(0.1)

        #self.w2 = nn.Linear( d_forward, d_shape)
        #self.drop2 = nn.Dropout(0.1)

    def forward(self, q, k, v):
        if False:
            print("multi head shape")
            print(q.shape, k.shape, v.shape)
        heads = torch.concat( [self.heads[i](q,k,v) for i in range(len(self.heads))], dim=2) #Used to be dim 0
        #m_attention = self.drop2( self.w2( self.drop1(self.w1(heads)) ) )
        m_attention = self.drop1(self.w1(heads)) 
        return m_attention


class decode_layer(nn.Module):
    def __init__(self, d_shape:int, d_forward:int=2048, heads:int=3):
        super().__init__()
        # Block 1
        self.mh_1 = multi_head( d_shape )
        self.ln_1 = nn.LayerNorm( d_shape )
       
        # Block 2
        self.mh_2 = multi_head( d_shape, d_forward, heads )
        self.ln_2 = nn.LayerNorm( d_shape )

        # Block 3
        self.w1 = nn.Linear( d_shape, d_forward )
        self.ln_3 = nn.LayerNorm( d_forward )
        
        # Block 4
        self.w2 = nn.Linear( (d_forward), d_shape )
        self.ln_4 = nn.LayerNorm( d_shape )


    def forward(self, in_encoding, out_encoding):
        block_1 = self.ln_1(
                    self.mh_1(out_encoding, out_encoding, out_encoding) + out_encoding
                  )  

        block_2 = self.ln_2(
                    self.mh_2(in_encoding, in_encoding, block_1) + out_encoding
                  )  

        block_3 = self.ln_3( self.w1( block_2 ) )
        block_4 = self.ln_4( self.w2( block_3 ) + block_2 )
        return block_4


class decoder(nn.Module):
    def __init__(self, d_shape, probs):
        super().__init__()
        # Both input and output vectors are the same shape
        self.d_layers = nn.ModuleList(
            [ decode_layer(d_shape) for i in range(6)]
        )

        self.w = nn.Linear(d_shape, probs)

    def forward(self, enc_v, out_v):
        x = out_v
        for f in self.d_layers:
            x = f( enc_v, x)
        #return F.softmax( self.w(x), dim=1 )
        return self.w(x).reshape(x.shape[0],1)


def transformer(device):
    #k = torch.rand(128,728).to(device, torch.half)
    vector_dim = 768
    return decoder(vector_dim, 1).to(device, torch.half)


