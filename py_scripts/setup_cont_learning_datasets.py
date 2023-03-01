# cd into the py_scripts directory before running this. 

import shutil 
import os
import CIFAR10_split_datasets
import MNIST_split_datasets
import CIFAR10_torchify
import CIFAR10_cached_data_split
import CIFAR10_cached_data_split_randomize
import compute_context_vector_MNIST 
import processing_MNIST 
from torchvision.datasets import MNIST, CIFAR10

if __name__ == '__main__':

    data_path = '../data/'

    # get the MNIST and CIFAR10 datasets.
    for data_func in [MNIST, CIFAR10]:
        # getting train and then test data!
        data_func(data_path, train=True, download=True)
        data_func(data_path, train=False,download=True)

    # process the MNIST data
    processing_MNIST.process_MNIST()
    
    split_dir = f'{data_path}splits/'
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    # moving the ConvMixerEmbeddings to the right folder. 
    shutil.copytree('../ConvMixerEmbeddings/', '../data/ConvMixerEmbeddings')

    # Splitting the ConvMixer embedded CIFAR10 data
    CIFAR10_cached_data_split.split_dataset()
    # making the other random seed splits too
    CIFAR10_cached_data_split_randomize.split_dataset()

    # This makes the raw data splits
    if not os.path.isdir(split_dir+'CIFAR10'):
        os.mkdir(split_dir+'CIFAR10')
    if not os.path.isdir(split_dir+'MNIST'):
        os.mkdir(split_dir+'MNIST')
    MNIST_split_datasets.split_dataset()
    CIFAR10_split_datasets.split_dataset()

    # torchifying the CIFAR dataset too (leads to double the loading speed!)
    CIFAR10_torchify.make_dataset()

    # making the context vectors for Active Dendrites: 
    compute_context_vector_MNIST.make_context_vectors()




