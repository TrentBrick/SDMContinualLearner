import torch 
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd 
import json 
import copy 
import pickle 
import os
import sys 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
    
# trying to make CIFAR10 loading faster.  

def data_processing(data, labels):
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    trans = transforms.ToTensor()
    # going from PIL to torch tensor converts things to floats as desired. 
    data = torch.stack( [(trans(Image.fromarray(d))*255).type(torch.uint8) for d in data])
    labels = torch.Tensor(labels).type(torch.int64) -1 # everything is 1 indexed?
    print("examples of the data and labels")
    print(data[:10], labels[:10])
    return data, labels

def make_dataset():
    
    data, labels = None, []
    # unify data from all of the different batches. 

    for i in range(10):
        print("loading in data batch:", i)
        t_dict = unpickle('../data/ImageNet32/train/train_data_batch_'+str(i+1))
        if data is None: 
            data = t_dict['data']
        else: 
            data = np.concatenate((data, t_dict['data']))
        # masking labels and then converting them to the coarse CIFAR versions
        labels += t_dict['labels']

    

    train_d, train_l = data_processing(data, labels)

    print("processed train data")

    # get the test data. 
    t_dict = unpickle('../data/ImageNet32/test/test_data')
    test_d, test_l = t_dict['data'], t_dict['labels']
    test_d, test_l = data_processing(test_d, test_l)

    print("processed test data")

    print("train and test data shapes", train_d.shape, test_d.shape, "train and test label sizes", len(train_l), len(test_l) )

    print("size of train data object", train_d.element_size() * train_d.nelement() )

    num_each_label_train_and_test = []
    for i in range(1000):
        num_each_label_train_and_test.append( ( (train_l==i).sum() , (test_l==i).sum() ) )

    print("Train and test appearances of each label type:",num_each_label_train_and_test )

    torch.save((train_d, train_l), "../data/ImageNet32/all_data_train.pt")
    torch.save((test_d, test_l), "../data/ImageNet32/all_data_test.pt")

if __name__ == '__main__':
    make_dataset() 

