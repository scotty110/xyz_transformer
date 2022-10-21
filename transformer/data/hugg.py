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
    def __init__(self, split:str='train', max_len:int=10):
        self.ds = load_dataset('wikitext', 'wikitext-103-v1', split=split)

        pretrained_model = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = self.tokenizer.vocab_size
        self.max_len = max_len

    def crop_text(self, txt):
        if len(txt)>1:
            t_split = txt.split(' ')
            if len(t_split) > 510:
                t_split = t_split[:510]
            return ' '.join(t_split)
        return txt

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index) -> tuple:
        # Format Text
        txt = self.ds[index]['text']

        #x_logits = self.tokenizer.encode(txt, max_length=self.max_len**2, padding='max_length', return_tensors='pt')
        x_logits = self.tokenizer.encode(txt, max_length=self.max_len**2, padding='max_length', truncation=True, return_tensors='pt')
        print(x_logits.shape)
        #x_logits = x_logits.squeeze() 
        x_logits = x_logits.reshape(self.max_len, self.max_len)
    
        return (x_logits )
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










