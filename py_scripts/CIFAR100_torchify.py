import torch 
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd 
import json 
import copy 
import pickle 
import os

save_data_path = "../data/CIFAR100/"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
# trying to make CIFAR10 loading faster.  

def data_processing(data, labels):
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    trans = transforms.ToTensor()
    # going from PIL to torch tensor converts things to floats as desired. Then back to uint8 for easier manipulation and more efficient compression. 
    data = torch.stack( [(trans(Image.fromarray(d))*255).type(torch.uint8) for d in data])
    labels = torch.Tensor(labels).type(torch.int64)
    print("examples of the data and labels")
    print(data[:10], labels[:10])
    return data, labels

def make_dataset():
    if not os.path.isdir(save_data_path):
        os.mkdir(save_data_path)

    for train_or_test in ['train', 'test']:

        t_dict = unpickle(f'../data/cifar-100-python/{train_or_test}')
        d, l = t_dict[b'data'], t_dict[b'fine_labels']

        d, l = data_processing(d, l)


        print(f"{train_or_test} data and label shapes", d.shape, l.shape)

        torch.save((d, l), save_data_path+f"all_data_{train_or_test}.pt")

if __name__ == '__main__':
    make_dataset() 

