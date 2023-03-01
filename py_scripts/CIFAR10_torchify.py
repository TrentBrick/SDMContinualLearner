import torch 
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd 
import json 
import copy 
import pickle 
import os

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
    if not os.path.isdir("../data/CIFAR10/"):
        os.mkdir("../data/CIFAR10/")
    data, labels = None, []
    # unify data from all of the different batches. 
    for i in range(5):
        t_dict = unpickle('../data/cifar-10-batches-py/data_batch_'+str(i+1))
        if data is None: 
            data = t_dict[b'data']
        else: 
            data = np.concatenate((data, t_dict[b'data']))
        labels += t_dict[b'labels']

    train_d, train_l = data_processing(data, labels)

    # get the test data. 
    t_dict = unpickle('../data/cifar-10-batches-py/test_batch')
    test_d, test_l = t_dict[b'data'], t_dict[b'labels']
    test_d, test_l = data_processing(test_d, test_l)

    print("train and test data shapes", train_d.shape, test_d.shape, "train and test label sizes", len(train_l), len(test_l) )

    num_each_label_train_and_test = []
    for i in range(10):
        num_each_label_train_and_test.append( ( (train_l==i).sum() , (test_l==i).sum() ) )

    print("Train and test appearances of each label type:",num_each_label_train_and_test )

    torch.save((train_d, train_l), "../data/CIFAR10/all_data_train.pt")
    torch.save((test_d, test_l), "../data/CIFAR10/all_data_test.pt")

if __name__ == '__main__':
    make_dataset() 

