'''
Test HuggingFace Dataloader for our transformer
https://huggingface.co/docs/datasets/v1.4.0/index.html
'''

from datasets import load_dataset
from transformers import pipeline, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch
import random


class wiki_loader():
    def __init__(self):
        self.ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')

        pretrained_model = 'bert-base-uncased'
        self.feature_extraction = pipeline('feature-extraction', model=pretrained_model, tokenizer=pretrained_model, truncation=True, device=0)

    def format_text(self, txt):
        if len(txt)>2:
            t_split = txt.split(' ')
            t_split = list(filter( lambda x: x != '', t_split))
            if len(t_split) > 512:
                print(txt)
                t_split = t_split[:512]
            return ' '.join(t_split)
        return txt


    def make_mask(self, txt:str) -> str:
        if len(txt)>2:
            t_split = txt.split(' ')
            #print(len(t_split))
            #t_split = list(filter( lambda x: x != '', t_split))
            #if len(t_split) > 512:
            #    t_split = t_split[:512]
            n = random.randrange(len(t_split))
            t_split[n] = '[MASK]'
            return ' '.join(t_split)
        return txt

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index) -> tuple:
        y_txt = self.format_text( self.ds[index]['text'] )
        x_txt = self.make_mask(y_txt)
        #print(y_txt, x_txt)
        y_ext = torch.Tensor( self.feature_extraction(y_txt) )
        x_ext = torch.Tensor( self.feature_extraction(x_txt) )

        if y_ext.shape[1] > 512:
            y_ext = y_ext[0][:512][:]
            x_ext = x_ext[0][:512][:]
    
        y = torch.zeros(1,512,768)
        x = torch.zeros(1,512,768)
       
        y[0][:y_ext.shape[1]][:] = y_ext
        x[0][:x_ext.shape[1]][:] = x_ext
    
        y = y.squeeze()
        x = x.squeeze()
        return (x, y, torch.Tensor([1]) )

t = wiki_loader()
t_dl = DataLoader( t, batch_size=128, shuffle=True,) 
x = iter(t_dl)
n_x = next(x)
while n_x != None:
    print(n_x[0].shape, n_x[2].shape)
    n_x = next(x)
print('Done')
