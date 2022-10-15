'''
Load Dataset and Model
'''
import time
import torch
import torch.nn as nn

from transformer.data import get_datasets
from transformer.model import get_model 

if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    train_dl, val_dl = get_datasets(64)

    '''
    Model, optimizer, loss func
    '''
    model = get_model(768)
    model = model.to(device, torch.half)

    opt = torch.optim.Adam( model.parameters(), 0.00001 )
    loss_func = nn.CrossEntropyLoss().to(device) #nn.MSELoss().to(device)

    start_t = time.time()

    tdl = iter(train_dl)

    epochs = 20
    for e in range(epochs):
        total_loss = 0
        while True:
            x,y,m = next(tdl)

            x = x.to(device, torch.half)
            y = y.to(device, torch.half)
            m = m.to(device, torch.half)

            opt.zero_grad()
            model.half()
            output = model(x,y)
            #print(output)
            loss = loss_func(output, m)
            total_loss += loss.item()
            print("Loss: {}".format(loss.item()))

            # Learn
            loss.backward()
            model.float()
            opt.step()
        print("Loss: {}".format(total_loss))
        print('Done Loop')

    '''
    while n_x != None:
        n_x = next(x)
    '''
    end_t = time.time()

    print("Elapsed Time: {}".format( end_t - start_t ) )


