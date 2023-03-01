import copy
from enum import Enum, auto
from types import SimpleNamespace
from typing import List
import numpy as np
import torch
from models import *
from torch import nn

from py_scripts import LightningDataModule
    
class DataSet(Enum):
    """
    Enum for the different datasets.
    """
    MNIST = auto()
    CIFAR10 = auto()
    CIFAR100 = auto()
    ImageNet32 = auto()
    SPLIT_MNIST = auto()
    SPLIT_CIFAR10 = auto()
    SPLIT_CIFAR100 = auto()
    ###
    Cached_ConvMixer_CIFAR10=auto()
    Cached_ConvMixer_ImageNet32=auto()
    Cached_ConvMixer_CIFAR100=auto()
    ###
    SPLIT_Cached_ConvMixer_CIFAR10 = auto()
    SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_3 = auto()
    SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_15 = auto()
    SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_27 = auto()
    SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_97 = auto()
    SPLIT_Cached_ConvMixer_ImageNet32=auto()
    SPLIT_CIFAR10_RandSeed_3 = auto()
    SPLIT_CIFAR10_RandSeed_15 = auto()
    SPLIT_CIFAR10_RandSeed_27 = auto()
    SPLIT_CIFAR10_RandSeed_97 = auto()
    SPLIT_Cached_ConvMixer_CIFAR100 = auto()
    

mnist_settings = dict(
    img_dim = 28,
    input_size = 784,
    nclasses = 10, 
    nchannels=1, 
    dataset_path_suffix='',
    split_path = 'MNIST/split_',
    num_data_splits = 5,
)

cifar10_settings = dict(
    img_dim = 32,
    input_size = 3072,
    nclasses = 10, 
    nchannels=3, 
    dataset_path_suffix = 'CIFAR10/all_data_',
    split_path = 'CIFAR10/split_',
    num_data_splits = 5,
)

cifar100_settings = dict(
    img_dim = 32,
    input_size = 3072,
    nclasses = 100, 
    nchannels=3, 
    dataset_path_suffix = 'CIFAR100/all_data_',
    split_path = 'CIFAR100/split_',
    num_data_splits = 50,
)

imagenet32_settings = dict(
    # inherit from cifar10
    img_dim = 32,
    input_size = 3072,
    nchannels=3, 
    nclasses = 1000,
    dataset_path_suffix = 'ImageNet32/all_data_',
    split_path = None,
    num_data_splits = 500,
)


######

cached_convmixer_cifar10_general_settings = dict(
    img_dim = None,
    input_size = 256,
    nclasses = 10, 
    nchannels=1, 
    num_data_splits=5,
)

cached_convmixer_cifar100_general_settings = dict(
    img_dim = None,
    input_size = 256,
    nclasses = 100, 
    nchannels=1, 
    num_data_splits=50,
)

cached_convmixer_cifar10_wtransforms_settings = dict(
    dataset_path_suffix = 'ConvMixerEmbeddings/CIFAR10/all_data_',
)

cached_convmixer_imagenet32_cifar10_wtransforms_settings = dict(
    dataset_path_suffix = 'ConvMixerEmbeddings/CIFAR10/all_data_',
    split_path = 'ConvMixerEmbeddings/CIFAR10/split_'
)

cached_convmixer_imagenet32_cifar100_wtransforms_settings = dict(
    dataset_path_suffix = 'ConvMixerEmbeddings/CIFAR100/all_data_',
    split_path = 'ConvMixerEmbeddings/CIFAR100/split_'
)

cached_convmixer_imagenet32_general_settings = dict(
    img_dim = None,
    input_size = 256,
    nclasses = 1000, 
    nchannels=1, 
    num_data_splits = 500,
)

cached_convmixer_imagenet32_wtransforms_settings = dict(
    dataset_path_suffix = 'ConvMixerEmbeddings/ConvMixerWTransforms_ImageNet32/all_data_',
)

cached_convmixer_imagenet32_imagenet32_wtransforms_settings = dict(
    dataset_path_suffix = 'ConvMixerEmbeddings/ImageNet32/all_data_',
)

def assign_dataset_params(dataset):
    dataset_params = {"dataset":dataset}
    if "MNIST" in dataset.name:
        dataset_params.update(mnist_settings)
    elif "CIFAR10" == dataset.name or "SPLIT_CIFAR10" == dataset.name:
        dataset_params.update(cifar10_settings)
    elif DataSet.SPLIT_CIFAR10.name in dataset.name and "RandSeed" in dataset.name:
        # modify the split path!
        dataset_params.update(cifar10_settings)
        randseed = dataset.name.split("RandSeed_")[-1]
        dataset_params['split_path'] = f'CIFAR10_RandSeed_{randseed}/split_'
    elif "CIFAR100" == dataset.name or "SPLIT_CIFAR100" == dataset.name:
        dataset_params.update(cifar100_settings)
    elif dataset.name == DataSet.ImageNet32.name:
        dataset_params.update(imagenet32_settings)
    elif dataset.name == DataSet.Cached_ConvMixer_CIFAR10.name or dataset.name == DataSet.SPLIT_Cached_ConvMixer_CIFAR10.name:
        dataset_params.update(cached_convmixer_cifar10_general_settings)
        dataset_params.update(cached_convmixer_imagenet32_cifar10_wtransforms_settings)
    elif DataSet.SPLIT_Cached_ConvMixer_CIFAR10.name in dataset.name and "RandSeed" in dataset.name:
        # modify the split path!
        dataset_params.update(cached_convmixer_cifar10_general_settings)
        dataset_params.update(cached_convmixer_imagenet32_cifar10_wtransforms_settings)
        randseed = dataset.name.split("RandSeed_")[-1]
        dataset_params['split_path'] = f'ConvMixerEmbeddings/CIFAR10_RandSeed_{randseed}/split_'
    elif DataSet.Cached_ConvMixer_CIFAR100.name == dataset.name or DataSet.SPLIT_Cached_ConvMixer_CIFAR100.name == dataset.name:
        # modify the split path!
        dataset_params.update(cached_convmixer_cifar100_general_settings)
        dataset_params.update(cached_convmixer_imagenet32_cifar100_wtransforms_settings)
    elif dataset.name == DataSet.Cached_ConvMixer_ImageNet32.name or dataset.name == DataSet.SPLIT_Cached_ConvMixer_ImageNet32.name:
        dataset_params.update(cached_convmixer_imagenet32_general_settings)
        dataset_params.update(cached_convmixer_imagenet32_imagenet32_wtransforms_settings)
    else:
        raise NotImplementedError(dataset.name)
    return dataset_params
