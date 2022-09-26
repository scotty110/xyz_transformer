import torch
from os.path import join

# Torch
import torch
import torch.nn as nn

# Custom Stuff
from ai4code.data.dataloader import get_dataloaders 
from ai4code.model.transformer import transformer
from ai4code.loops.train import train_loop 
from ai4code.loops.building import build_loop 

if __name__ == '__main__':
    # Torch Device
    device = torch.device('cuda:0')

    # Data loader
    data_dir='/home/squirt/Documents/AI4Code'
    emb_file = join(data_dir, 'embeddings_small_f16.pkl') 
    key_file = join(data_dir, 'embeddings_small_keys.pkl')
  
    t_dl, e_dl = get_dataloaders(emb_file, key_file)
    print('Loaded Data')

    # Get Model, loss function and optimizer
    model = transformer(device)
    optim = torch.optim.Adam( model.parameters(), 0.00002)
    loss_func = nn.MSELoss().to(device)

    # Test Train
    train_loop(t_dl, e_dl, model, loss_func, optim, device, 2000)


    print('Done')

