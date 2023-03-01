
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch 
import pandas as pd
from torchvision.io import read_image

# TODO: should add the params modifications in params.py here that are relevant to the dataset. 
class SPLIT_Dataset(Dataset):
    def __init__(self, dataset_str, split_index, split_path, transform=None, target_transform=None, train=True):
        # datasets to be located in eg. ../data/splits/CIFAR10_1.pt indexed by their split (there are 5 of them)
        master_loc = 'data/splits/'
        self.split_dir = master_loc+split_path 
        self.dataset_str = dataset_str

        # this is a directory with all of the images inside stored as
        train_or_test = 'train' if train else "test"
        self.split_dir += train_or_test+"_"
        self.img_and_label_dir = self.split_dir+ str(split_index)+".pt"
        # loading all the images here as a large torch tensor rather than doing it one by one. 
        try: 
            self.images, self.img_labels = torch.load(self.img_and_label_dir)
        except:
            self.images, self.img_labels = torch.load("../"+self.img_and_label_dir)
        
        #self.label_path = self.split_dir+ str(split_index)+'_labels'+'.csv'

        #self.img_labels = pd.read_csv(self.label_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.images[idx] #[read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if "MNIST" in self.dataset_str: # could do this on back end but would take much more memory
            image = image.type(torch.float)

        if image.dtype is torch.uint8:
            # more efficient form of storage
            image = image.type(torch.float)/255

        return image, label

class Torch_Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None, train=True, download=None):
        # download is just to be compatible with other functions. 
        # stored as a tuple of data and labels. Already processed as a torch tensor. 
        self.dataset_path = dataset_path

        # this is a directory with all of the images inside stored as
        train_or_test = 'train' if train else "test"
        self.img_and_label_dir = dataset_path+ train_or_test+".pt"
        
        # loading all the images here as a large torch tensor rather than doing it one by one. 
        self.images, self.img_labels = torch.load(self.img_and_label_dir)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #print(idx, self.images.shape, self.images[idx])
        
        image = self.images[idx] #[read_image(img_path)
        label = self.img_labels[idx]
        # TODO: store as int8 and then divide by 256 here. 
        
        if "MNIST" in self.dataset_path: # could do this on back end but would take much more memory
            image = image.type(torch.float)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        if image.dtype is torch.uint8:#"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
            image = image.type(torch.float)/255

        return image, label

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, params, data_path="data/" ):
        super().__init__()
        self.batch_size = params.batch_size
        self.num_workers = params.num_workers
        self.data_path = data_path
        # these are for continual learning: 
        self.continual_learning=params.continual_learning
        self.curr_index = -1 # will increment right away on the first training run. 
        self.epochs_per_dataset = params.epochs_per_dataset
        self.dataset_str = params.dataset_str
        if self.continual_learning:
            self.num_data_splits = params.num_data_splits
            self.split_path = params.split_path

        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
            
        transforms_list = []
        # small amount of special processing here. 
        if "MNIST" in self.dataset_str: 
            # everything is already being saved as a Torch tensor as it is a custom dataset.    
            if "SPLIT" in params.dataset_str:
                # TODO: make this more concise. 
                self.data_function = SPLIT_Dataset
            else: 
                self.data_function = MNIST
                transforms_list.append(transforms.ToTensor())
            transforms_list.append( 
                    transforms.Lambda(lambda x: x.type(torch.float)/255 )
                )
                
            if params.normalize_n_transform_inputs:
                transforms_list.append( transforms.Normalize((0.5, ), (0.5,)) )
        else:
            if "SPLIT" in params.dataset_str:
                self.data_function = SPLIT_Dataset
            else: 
                self.data_function = Torch_Dataset

        if "CIFAR10" in params.dataset_str and params.normalize_n_transform_inputs:
                transforms_list.append( transforms.Normalize(cifar10_mean, cifar10_std) )

        if params.dataset_str == "MNIST" and not params.normalize_n_transform_inputs:
            if params.adversarial_attacks:
                # adding an epsilon so its non zero for adversarial attacks to be possible
                print("adding lambda transform to made the mnist adversary be able to attack everything!!!")  
                transforms_list.append(transforms.Lambda(lambda x: torch.where(x==0.0, x+0.01, x) ))

        if params.use_convmixer_transforms and not params.normalize_n_transform_inputs:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(params.scale, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=params.ra_n, magnitude=params.ra_m),
                transforms.ColorJitter(params.jitter, params.jitter, params.jitter),
                #transforms.ToTensor(),
                # the input is unint8. Then but it is already a tensor not a PIL so need to convert to float and decimal points
                transforms.Lambda(lambda x: x.type(torch.float)/255 ),
                transforms.Normalize(cifar10_mean, cifar10_std),
                transforms.RandomErasing(p=params.reprob)
            ])

            self.test_transform = transforms.Compose([
                #transforms.ToTensor(),
                transforms.Lambda(lambda x: x.type(torch.float)/255 ),
                transforms.Normalize(cifar10_mean, cifar10_std)
            ])

        else: 
            self.train_transform = transforms.Compose(transforms_list)
            self.test_transform = transforms.Compose(transforms_list)

    def setup(self, stage, train_shuffle=True, test_shuffle=True):
        
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        
        if self.continual_learning: 
            # will make datasets from each of the classes.
            self.train_datasets = [self.data_function(self.dataset_str, i,  self.split_path, transform=self.train_transform, train=True) for i in range(self.num_data_splits)] # assuming for now there are 5 splits for both datasets. 
            self.test_datasets = [self.data_function(self.dataset_str, i, self.split_path, transform=self.test_transform, train=False) for i in range(self.num_data_splits)]

        else: 
            self.train_data = self.data_function(
                self.data_path, train=True, download=True, transform=self.train_transform
            )
            self.test_data = self.data_function(
                self.data_path, train=False,download=True, transform=self.test_transform
            )

    def train_dataloader(self):
        # updating here, called before validation dataloader: 
        if self.continual_learning and self.trainer.current_epoch % self.epochs_per_dataset == 0:
            if self.curr_index<(self.num_data_splits-1): # 5 datasets for now.
                # TODO: make this more general. 
                self.curr_index += 1
                print("SWITCHING DATASET BY INCREMENTING INDEX TO:", self.curr_index, "EPOCH IS:", self.trainer.current_epoch)

        if self.continual_learning: 
            self.train_data = self.train_datasets[self.curr_index]

        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):

        print("RELOADING VAL DATALOADER")

        if self.continual_learning:
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=True , num_workers=self.num_workers) for ds in self.test_datasets[:self.curr_index+1]]
        else: 
            return DataLoader(
                self.test_data, batch_size=self.batch_size, shuffle=self.test_shuffle , num_workers=self.num_workers
            )