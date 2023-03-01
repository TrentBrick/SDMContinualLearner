import torch 
import numpy as np
import os

# for saving the data
cached_model_path= "MNIST"

def get_context(data, labels, save_name):
    print("Save name data split is:", save_name)
    contexts = []
    for l in range(10):

        lmask = torch.where(labels==l, 1,0)
        indices = lmask.nonzero().squeeze()
        temp_d = data[indices]
        print(temp_d.shape)
        temp_d = temp_d.sum(0)/len(temp_d)
        print(temp_d.shape)
        print(temp_d, l)
        temp_d = temp_d.type(torch.float).flatten()
        contexts.append(temp_d)

    contexts = torch.stack(contexts)
    print(contexts.shape)

    torch.save(contexts, f"../data/context_vectors/{cached_model_path}.pt")

def make_context_vectors():
    for save_name in ['train']:
        loaded_data, loaded_labels =torch.load('../data/MNIST/processed/training.pt')

        if not os.path.isdir('../data/context_vectors/'):
            os.mkdir('../data/context_vectors/')
        get_context(loaded_data, loaded_labels, save_name)