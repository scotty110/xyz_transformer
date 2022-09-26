'''
Make transformer parts
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''

import math
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


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
