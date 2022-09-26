'''
Make a data loader for training:
    - Will have 2 examples, swap dataset and consecutive dataset
'''

import numpy as np
import torch
import pickle

from torch.utils.data import Dataset, DataLoader
from random import shuffle


class consecutive_cells(Dataset):
    '''
    This dataset is going to be slightly different in it will generate "new"
    data after each epoch. 
    '''
    def __init__( self,
                    file:str,
                    rm_keys:list,
                    num_pos:int=1,
                    num_neg:int=3):
        # Load Data
        self.file_name = file
        self.data = self.load_data(rm_keys)
    
        # Generate training pairs
        self.num_pos = num_pos
        self.num_neg = num_neg
        if self.num_neg < 1 or self.num_pos < 1:
            raise Exception("num pos/neg needs to be greater than 2 to form a +- pair")
        self.training_data = self.select_data()
        return

    def load_data(self, rm_keys:list)->dict:
        # Just load Pickel file
        with open(self.file_name, 'rb') as handle:
            data = pickle.load(handle) 
        # "split" dataset
        for k in rm_keys: 
            del data[k]
        return data

    def refresh(self):
        self.training_data = self.select_data()
        return

    def select_data(self)->list:
        '''
        Generate pairs (arr_key, input_pos, output_pos, prob)
        '''
        pair_list = [ None for i in range( len(self.data)*(self.num_pos + self.num_neg))]
        pair_count = 0 # counter for pair_list
        for arr_ID,arr in self.data.items():
            # Generate Positive examples
            for i in range(self.num_pos):
                r1 = np.random.randint(0,arr.shape[0])
                r2 = r1+1
                if r2 >= arr.shape[0]:
                    r2= -1 
                pair_list[pair_count] = (arr_ID, r1, r2, 1.) #TODO
                pair_count += 1

            # Generate Negative Examples
            for i in range(self.num_neg):
                # Generate Random pairs
                r1 = np.random.randint(0,arr.shape[0])  
                r2 = np.random.randint(0,arr.shape[0])  
                while r1 == r2:
                    r2 = np.random.randint(0,arr.shape[0])  
                pair_list[pair_count] = (arr_ID, r1, r2, 0.) #TODO
                pair_count += 1
        return pair_list


    def __len__(self):
        return len(self.training_data)

    def select_pair(self, pair):
        pass

    def __getitem__(self,index)->tuple:
        '''
        Preforms the actual get data
        Inputs:
            - index (int): The index in self.training_data to return
        Returns:
            A tuple in the format (X,y) where X is a 2xSize tensor, y=0/1
        '''
        a_id, p1, p2, prob = self.training_data[ index ]
        all_arr = self.data[a_id]
        # Select Data
        in_x = all_arr[p1]
        in_x = torch.reshape(in_x, (1,*in_x.shape))
        if p2 == -1:
            out_x = torch.zeros(in_x.shape) 
        else:
            out_x = all_arr[p2]
            out_x = torch.reshape(out_x, (1,*out_x.shape))

        return (in_x, out_x, prob)


# Could replace loader in class above -- TODO
def load_pkl_file(file_path:str):
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle) 
    return data


def get_dataloaders(emb_file:str, key_file:str, tv_split:float=.7)->tuple:
    '''
    Create 2 dataloaders (from the embedding datasets), Eval and a Train.
    Paritioning the keys random each time they are created
    Inputs:
        - emb_file (str): Path to the embedding dict
        - key_file (str): Path to the key list
        - tv_split (float): % of keys to use as training data (1-x for eval).
    Returns:
        Tuple of training and eval dataloaders

    TODO -- Add more variable passthough
    '''
    # Get and partition keys
    keys = load_pkl_file(key_file)
    shuffle(keys)
    #print('# keys: {}'.format(len(keys)))
    x = int(len(keys) * tv_split)
    e_keys = keys[:x] # If key is present del it from dataset
    t_keys = keys[x:]
    #print('x({}) t({}) e({})'.format(x,len(t_keys), len(e_keys)))

    # Create Datasets
    t_ds = consecutive_cells(emb_file, t_keys) 
    e_ds = consecutive_cells(emb_file, e_keys) 

    # Create Dataloaders
    # Caused memory to overflow
    #t_dl = DataLoader( t_ds, batch_size=64, pin_memory=True, shuffle=True, num_workers=6,)
    #e_dl = DataLoader( e_ds, batch_size=64, pin_memory=True, shuffle=True, num_workers=6,)

    t_dl = DataLoader( t_ds, batch_size=2048, shuffle=True, num_workers=6,)
    e_dl = DataLoader( e_ds, batch_size=2048, shuffle=True, num_workers=6,)

    return (t_dl, e_dl) 







