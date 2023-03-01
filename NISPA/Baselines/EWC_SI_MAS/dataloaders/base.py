import torchvision
import torch.utils
from torchvision import transforms
from .wrapper import CacheClassLabel


#Python cannot pickle lambda function
#Give named function instead
def add10(y):
    return y + 10
def add47(y):
    return y + 47

def TASK5(dataroot, train_aug=False, seed = 0):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_transform = test_transform

    
    train_dataset_EMNIST = torchvision.datasets.EMNIST(
        root=dataroot,
        split="balanced",
        train=True,
        download=True,
        transform=train_transform
    )
    
    train_dataset_FMNIST = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform,
        target_transform = add47
    )
    
    
    #Val Datasets
    
    train_dataset_EMNIST, val_dataset_EMNIST = torch.utils.data.random_split(train_dataset_EMNIST, [ int(len(train_dataset_EMNIST)*0.9),   
                                                        len(train_dataset_EMNIST) - int(len(train_dataset_EMNIST)*0.9)])
    
    train_dataset_FMNIST, val_dataset_FMNIST = torch.utils.data.random_split(train_dataset_FMNIST, [ int(len(train_dataset_FMNIST)*0.9),   
                                                        len(train_dataset_FMNIST) - int(len(train_dataset_FMNIST)*0.9)])
    
    #Test Datasets

    test_dataset_EMNIST = torchvision.datasets.EMNIST(
        root=dataroot,
        split="balanced",
        train=False,
        download=True,
        transform=test_transform
    )
    test_dataset_FMNIST = torchvision.datasets.FashionMNIST(
        root=dataroot,
        train=False,
        download=True,
        transform=test_transform,
        target_transform = add47
    )
    
    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_EMNIST, train_dataset_FMNIST])
    train_dataset.root = dataroot
    train_dataset = CacheClassLabel(train_dataset, "TASK5", seed)
    
    
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_EMNIST, val_dataset_FMNIST])
    val_dataset.root = dataroot
    val_dataset = CacheClassLabel(val_dataset, "TASK5", seed)
    
    
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_EMNIST, test_dataset_FMNIST])
    test_dataset.root = dataroot
    test_dataset = CacheClassLabel(test_dataset, "TASK5", seed)

    return train_dataset, val_dataset, test_dataset




def CIFAR10(dataroot, train_aug=False, seed = 0):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_transform = val_transform

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [ int(len(train_dataset)*0.9),   
                                                        len(train_dataset) - int(len(train_dataset)*0.9)])
    
    train_dataset.root = dataroot
    train_dataset = CacheClassLabel(train_dataset, "CIFAR10", seed)
    val_dataset.root = dataroot
    val_dataset = CacheClassLabel(val_dataset, "CIFAR10", seed)
    


    test_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    test_dataset.root = dataroot
    test_dataset = CacheClassLabel(test_dataset, "CIFAR10", seed)
    return train_dataset, val_dataset, test_dataset


def CIFAR100(dataroot, train_aug=False, seed = 0):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_transform = val_transform



    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [ int(len(train_dataset)*0.9),   
                                                        len(train_dataset) - int(len(train_dataset)*0.9)])
    
    train_dataset.root = dataroot
    train_dataset = CacheClassLabel(train_dataset, "CIFAR100", seed)
    val_dataset.root = dataroot
    val_dataset = CacheClassLabel(val_dataset, "CIFAR100", seed)
    
    
    

    test_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    test_dataset.root = dataroot
    test_dataset = CacheClassLabel(test_dataset, "CIFAR100", seed)
    

    return train_dataset, val_dataset, test_dataset

