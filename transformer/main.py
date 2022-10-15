'''
Load Dataset and Model
'''
import time

from transformer.data import get_datasets
from transformer.model import get_model 

if __name__ == '__main__':
    train_dl, val_dl = get_datasets(512)
    start_t = time.time()
    x = iter(train_dl)
    n_x = next(x)
    while n_x != None:
        n_x = next(x)
    end_t = time.time()

    print("Elapsed Time: {}".format( end_t - start_t ) )
    print('Done')


