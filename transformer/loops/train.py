'''
Make training loop for
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def train_loop(  
                t_dl:DataLoader,
                e_dl:DataLoader,
                model:nn.Module,
                loss_func:nn.Module,
                opt:torch.optim,
                device:torch.device,
                loops:int = 5
                ):
    '''
    One Training loop, n trains then 1 eval
    Inputs:
        ...
    '''
    for i in range(loops):
        t_dl.dataset.refresh()
        t_iter = iter(t_dl)
        loop_loss = 0.

        for x in t_iter:
            in_x = x[0].to(device, torch.half) 
            out_x = x[1].to(device, torch.half) 
            probs = x[2].to(device, torch.half)
            probs = torch.reshape(probs, (*probs.shape, 1))

            opt.zero_grad()
            model.half()
            output = model(in_x, out_x)                                                  
            loss = loss_func(output, probs)                                             
            loop_loss += loss.item() # Need to log/graph later
  
            # "Learn"                                                               
            #opt.zero_grad()
            loss.backward()                                                         
            model.float()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()                                                        

        print("avg loss: {}".format( loop_loss/len(t_dl)))
        '''
            break
        break
        #'''


    print('done')

