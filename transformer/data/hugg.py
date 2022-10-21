'''
Test HuggingFace Dataloader for our transformer
https://huggingface.co/docs/datasets/v1.4.0/index.html
'''

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import random


class wiki_loader():
    def __init__(self, split:str='train'):
        self.ds = load_dataset('wikitext', 'wikitext-103-v1', split=split)

        pretrained_model = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index) -> tuple:
        # Format Text
        txt = self.ds[index]['text']
        x_tokens = self.tokenizer.tokenize(txt)
        #x_logits = self.tokenizer.encode(x_tokens)
        x_logits = self.tokenizer.encode(txt)
        x_decode = self.tokenizer.decode(x_logits)
        d_tokens = self.tokenizer.tokenize(x_decode)

        # AutoFill
        #y = torch.zeros(1,512,768)
       
        #y[0][:y_ext.shape[1]][:] = y_ext
    
        #y = y.squeeze()
        #return (x_tokens)
        return len(d_tokens)-2 == len(x_tokens)
'''
def get_datasets(batch_size):
    train_dl = DataLoader( wiki_loader('train'), batch_size=batch_size, shuffle=True )
    val_dl = DataLoader( wiki_loader('validation'), batch_size=batch_size, shuffle=True )
    return (train_dl, val_dl)
'''

if __name__ == '__main__':
    test_ds = wiki_loader()
    for i in range(100):
        print(test_ds[i])










