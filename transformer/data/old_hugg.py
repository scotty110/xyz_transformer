'''
Test HuggingFace Dataloader for our transformer
https://huggingface.co/docs/datasets/v1.4.0/index.html
'''
from datasets import load_dataset_builder
from datasets import load_dataset

#from transformers import AutoTokenizer

ds_builder = load_dataset_builder('wikitext', 'wikitext-103-v1')
#ds_builder = load_dataset_builder("wikitext-103-v1")
#ds_builder = load_dataset_builder("rotten_tomatoes")

#print(ds_builder.info.description)
#print(ds_builder.info.features)


raw_ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
ds = raw_ds.filter(lambda x: x != "")
print(ds[0])
print(ds[1])
print(len(ds))
print(len(raw_ds))



'''
Tokenizer
https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
'''
from transformers import BertModel, BertTokenizer
import torch

# Initialize the tokenizer with a pretrained model
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#embedder = BertModel.from_pretrained("bert-base-uncased")

test_text = "granola bars"

tokens = tokenizer.basic_tokenizer.tokenize(test_text)
ids = torch.tensor(tokenizer.encode(tokens)).unsqueeze(0)
print( tokenizer("granola bars") )
print(ids)
'''

'''
from transformers import pipeline, AutoTokenizer

# direct encoding of the sample sentence
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
encoded_seq = tokenizer.encode("i am sentence")

# your approach
feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base")
features = feature_extraction("i am sentence")

# Compare lengths of outputs
print(len(encoded_seq)) # 5
print(encoded_seq)
# Note that the output has a weird list output that requires to index with 0.
print(len(features[0])) # 5
print(len(features[0][1]))
'''
from transformers import pipeline 

pretrained_model = 'bert-base-uncased'
feature_extraction = pipeline('feature-extraction', model=pretrained_model, tokenizer=pretrained_model)
features = feature_extraction("Hello World")
print(len(features[0]))
print(len(features[0][1]))



print('Done')
