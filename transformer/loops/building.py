'''
Make training loop for
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def build_loop(  model:nn.Module,
                    loss:nn.Module,
                    opt:torch.optim,
                    device:torch.device,
                    ):
   
    in_x = torch.zeros(128, 1, 768).to(device, torch.half) 
    out_x = torch.zeros(128, 1, 768).to(device, torch.half) 
    probs = torch.zeros(128, 1).to(device, torch.half)
    print( in_x.shape, out_x.shape, probs.shape)

    output = model(in_x, out_x)                                                  
    loss = loss(output, probs)                                             
    print(loss.item())  # Need to log/graph later
  
    # "Learn"                                                               
    opt.zero_grad()
    loss.backward()                                                         
    opt.step()                                                        


    print('done')

