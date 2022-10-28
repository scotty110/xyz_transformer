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

    def make_mask(self, txt:str) -> str:                                        
        if len(txt)>2:                                                          
            t_split = txt.split(' ')                                            
            n = random.randrange(len(t_split))                                  
            t_split[n] = '[MASK]'                                               
            return ' '.join(t_split)                                            
        return txt

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index) -> tuple:
        # Format Text
        y_txt = self.ds[index]['text']
        x_txt = self.make_mask(y_txt)

        y_logits = self.tokenizer.encode(y_txt, max_length=self.max_len**2, padding='max_length', truncation=True, return_tensors='pt')
        y_logits = x_logits.reshape(self.max_len, self.max_len)
        
        x_logits = self.tokenizer.encode(x_txt, max_length=self.max_len**2, padding='max_length', truncation=True, return_tensors='pt')
        x_logits = x_logits.reshape(self.max_len, self.max_len)

        probs = torch.Tensor([1,0])
    
        return (x_logits, y_logits, probs )

def get_datasets(batch_size):
    train_dl = DataLoader( wiki_loader('train'), batch_size=batch_size, shuffle=True )
    val_dl = DataLoader( wiki_loader('validation'), batch_size=batch_size, shuffle=True )
    return (train_dl, val_dl)
'''

if __name__ == '__main__':
    test_ds = wiki_loader()
    for i in range(100):
        print(test_ds[i])
'''









