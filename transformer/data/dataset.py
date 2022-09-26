from datasets import load_dataset, load_dataset_builder

ds_name = 'bookcorpus'

ds_builder = load_dataset_builder( ds_name )
#print(ds_builder.info.description)
#print( ds_builder.info.features )

ds = load_dataset( ds_name )
print(ds)

print(ds['train'][0])
print(type(ds))
