'''
Load Dataset and Model
'''
import time
import torch

from transformer.data import get_datasets
from transformer.model import get_model 

if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    train_dl, val_dl = get_datasets(64)

    model = get_model(768)
    model = model.to(device, torch.half)

    start_t = time.time()
    '''
    tdl = iter(train_dl)
    tdl_t = next(tdl)
   
    x,y,m = tdl_t

    x = x.to(device, torch.half)
    y = y.to(device, torch.half)
    m = m.to(device, torch.half)
    '''
    x = torch.ones(64,512,768).to(device, torch.half)
    y = torch.ones(64,512,768).to(device, torch.half)
    m = torch.ones(64,1).to(device, torch.half)
    
    m_a = model(x,y) 


    '''
    while n_x != None:
        n_x = next(x)
    '''
    end_t = time.time()

    print("Elapsed Time: {}".format( end_t - start_t ) )
    print('Done')


