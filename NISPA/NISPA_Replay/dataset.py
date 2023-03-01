from torchvision.datasets import  MNIST as MNIST_TORCH
from continuum.datasets import MNIST,  FashionMNIST, EMNIST, Fellowship, CIFAR100, CIFAR10
from continuum.datasets import InMemoryDataset
import torch
import numpy as np
from continuum import ClassIncremental
from torchvision import transforms, utils

def get_transformers(dataset_name):

    if dataset_name == "cifar10":
        transform_train = [transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])]

        transform_test = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])]
    else:
        transform_train = None
        transform_test = None
    return transform_train, transform_test


def get_dataset(dataset, increment, args = None):
    transform_train, transform_test = get_transformers(dataset)
    if dataset == 'mnist':
        train_dataset = MNIST('Data', train = True, download=True)
        test_dataset = MNIST('Data', train = False, download=True)
        input_dim = 28*28
     
    if dataset == 'fmnist':
        train_dataset = FashionMNIST('Data', train = True, download=True)
        test_dataset = FashionMNIST('Data', train = False, download=True)
        input_dim  = 28*28
    if dataset == 'emnist':
        train_dataset = EMNIST('Data', train = True, download=True, split='balanced')
        test_dataset = EMNIST('Data', train = False, download=True, split='balanced')
        input_dim  = 28*28

    if dataset=="cifar10":
        train_dataset = CIFAR10('Data', train = True, download=True)
        test_dataset = CIFAR10('Data', train = False, download=True)
        input_dim = 3

    if dataset == "cifar100":
        train_dataset = CIFAR100('Data', train = True, download=True)
        test_dataset = CIFAR100('Data', train = False, download=True)
        input_dim = 3
    
        
    if dataset == "emnist_fmnist":
        train_dataset = Fellowship([EMNIST('Data', train = True, download=True, split='balanced'), 
                                    FashionMNIST('Data', train = True, download=True)])
        
        test_dataset = Fellowship([EMNIST('Data', train = False, download=True, split='balanced'), 
                                   FashionMNIST('Data', train = False, download=True)])
        
        scenario_train = ClassIncremental(train_dataset, increment=[10, 13, 13, 11, 10], transformations = transform_train)
        scenario_test = ClassIncremental(test_dataset, increment=[10, 13, 13, 11, 10], transformations = transform_train)
        output_dim = max(train_dataset.get_data()[1]) + 1
        input_dim  = 28*28
        return scenario_train,  scenario_test, input_dim, output_dim, transform_train, transform_test
        
    output_dim = max(train_dataset.get_data()[1]) + 1
    
    scenario_train = ClassIncremental(train_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment, transformations = transform_train)
    scenario_test = ClassIncremental(test_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment, transformations = transform_test)
    
    
    return scenario_train, scenario_test, input_dim, output_dim, transform_train, transform_test
