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
        self.w1 = nn.Linear((heads*d_shape), d_shape)
        self.drop1 = nn.Dropout(0.1)

    def forward(self, q, k, v):
        if False:
            print("multi head shape")
            print(q.shape, k.shape, v.shape)
        heads = torch.concat( [self.heads[i](q,k,v) for i in range(len(self.heads))], dim=2) #Used to be dim 0
        #m_attention = self.drop2( self.w2( self.drop1(self.w1(heads)) ) )
        m_attention = self.drop1(self.w1(heads)) 
        return m_attention

