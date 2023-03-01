
import torch 
import numpy as np
import os

cached_path = "CachedOutputs"
# for saving the data
cached_model_path= "ConvMixerWTransforms_ImgNet32_CIFAR10"
num_data_splits =5

def get_context(data, labels, save_name):
    print("Save name data split is:", save_name)
    contexts = []
    for l in range(num_data_splits*2):

        lmask = torch.where(labels==l, 1,0)
        indices = lmask.nonzero().squeeze()
        temp_d = data[indices]
        print(temp_d.shape)
        temp_d = temp_d.sum(0)/len(temp_d)
        print(temp_d.shape)
        print(temp_d, l)
        contexts.append(temp_d)

    contexts = torch.stack(contexts)
    print(contexts.shape)

    torch.save(contexts, f"../data/context_vectors/{cached_model_path}.pt")

for save_name in ['train']:
    loaded_data, loaded_labels = torch.load(f"../data/{cached_path}/{cached_model_path}/all_data_{save_name}.pt")
    get_context(loaded_data, loaded_labels, save_name)