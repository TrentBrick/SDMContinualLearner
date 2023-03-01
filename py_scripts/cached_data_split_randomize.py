
import torch 
import numpy as np
import os
import random

def get_label_matches(l1,l2, d, l):
    lmask = torch.where(torch.logical_or(l==l1, l==l2), 1,0)
    indices = lmask.nonzero().squeeze()
    print(indices )
    return d[indices], l[indices]

def split_data(data, labels, save_name):
    print("Save name data split is:", save_name)
    
    for i in range(num_data_splits):
        l1_ind = i*2
        l2_ind = l1_ind+1
        l1, l2 = labels_ordered[l1_ind], labels_ordered[l2_ind]
        print("Labels used here:",l1,l2)

        matched_data, matched_labels = get_label_matches(l1,l2, data, labels)

        print(matched_data.shape, len(matched_labels), matched_labels.shape)

        torch.save((matched_data, matched_labels), f"{save_dir_path}/split_{save_name}_{str(i)}.pt")

if __name__ == '__main__':

    cached_path = "CachedOutputs"
    # for saving the data
    cached_model_path_load= "ConvMixerWTransforms_ImgNet32_CIFAR10"
    num_data_splits =5

    seeds = [3,15,27,97]
    for seed in seeds: 
        print("==="*10)
        print("Seed is:",seed)
        print("==="*10)

    cached_model_path_save= f"ConvMixerWTransforms_ImgNet32_CIFAR10_RandSeed_{seed}"

    save_dir_path = f"../data/splits/{cached_path}"
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    save_dir_path+= "/"+cached_model_path_save
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    labels_to_randomize = np.arange(num_data_splits*2)
    labels_ordered = np.random.choice(np.arange(num_data_splits*2), num_data_splits*2, replace=False)

    for save_name in ['train', 'test']:
        loaded_data, loaded_labels = torch.load(f"../data/{cached_path}/{cached_model_path_load}/all_data_{save_name}.pt")
        split_data(loaded_data, loaded_labels, save_name)