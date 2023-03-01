
import torch 
import numpy as np
import os

cached_path = "ConvMixerEmbeddings"
# for saving the data
cached_model_path= "CIFAR10"
num_data_splits =5


def get_label_matches(l1,l2, d, l):
    lmask = torch.where(torch.logical_or(l==l1, l==l2), 1,0)
    indices = lmask.nonzero().squeeze()
    print(indices )
    return d[indices], l[indices]

def split_data(data, labels, save_name, save_dir_path):
    print("Save name data split is:", save_name)
    for i in range(num_data_splits):
        l1 = i*2
        l2 = l1+1

        matched_data, matched_labels = get_label_matches(l1,l2, data, labels)

        print(matched_data.shape, len(matched_labels), matched_labels.shape)

        torch.save((matched_data, matched_labels), f"{save_dir_path}/split_{save_name}_{str(i)}.pt")


def split_dataset():

    save_dir_path = f"../data/splits/{cached_path}"

    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    save_dir_path+= "/"+cached_model_path
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)


    for save_name in ['train', 'test']:
        loaded_data, loaded_labels = torch.load(f"../data/{cached_path}/{cached_model_path}/all_data_{save_name}.pt")
        split_data(loaded_data, loaded_labels, save_name, save_dir_path)