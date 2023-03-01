from torchvision import datasets, transforms
from data.manipulate import UnNormalize
import numpy as np


# Specify available data-sets
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'fashionmnist': datasets.FashionMNIST
}


# Specify available transforms
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'fashionmnist': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'cifar10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'cifar100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'cifar10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}


# Specify configurations of available data-sets
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'combined': {'size': 28, 'channels': 1, 'classes': 20},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'mnist_combined': {'size':10, 'channels':1, 'classes': 20},
    'precifar100': {'size':23, 'channels':1, 'classes': 100}
}