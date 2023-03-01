from torchvision.datasets import  MNIST as MNIST_TORCH
from continuum.datasets import MNIST,  FashionMNIST, EMNIST, Fellowship, CIFAR100, CIFAR10
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from continuum import ClassIncremental
from continuum.datasets import InMemoryDataset
from torchvision.transforms import transforms

def get_dataset(dataset, increment=2, args = None):
    
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
        
    if dataset == "pmnist_partial":
        return get_permuted_MNIST(args.perm_perc)

        
    if dataset == "emnist_fmnist":
        train_dataset = Fellowship([EMNIST('Data', train = True, download=True, split='balanced'), 
                                    FashionMNIST('Data', train = True, download=True)])
        
        test_dataset = Fellowship([EMNIST('Data', train = False, download=True, split='balanced'), 
                                   FashionMNIST('Data', train = False, download=True)])
        
        scenario_train = ClassIncremental(train_dataset, increment=[10, 13, 13, 11, 10])
        scenario_test = ClassIncremental(test_dataset, increment=[10, 13, 13, 11, 10])
        output_dim = max(train_dataset.get_data()[1]) + 1
        input_dim  = 28*28
        return scenario_train,  scenario_test, input_dim, output_dim

    if "split_" in dataset:

        dataset_str = dataset.split("_")[-1]

        if dataset_str == 'cifar100':
            input_dim = 256
            num_data_splits = 50
            output_dim = 100
            data_path = 'Data/CachedOutputs/ConvMixerWTransforms_ImgNet32_CIFAR100/all_data_'
        
        elif dataset_str == 'cifar10':
            input_dim = 256
            num_data_splits = 5
            output_dim = 10
            data_path = 'Data/CachedOutputs/CIFAR10/all_data_'
        
        elif dataset_str == 'mnist':
            input_dim = 28*28
            num_data_splits = 5
            output_dim = 10

        increment = 2

        train_dataset = InMemoryDataset(*give_torch_dataset(
                data_path, True))
        test_dataset = InMemoryDataset(*give_torch_dataset(
                data_path, False))

        
        '''train_dataset = InMemoryDataset(Torch_Dataset(
                data_path, train=True, download=True, transform=None
            ))
        test_dataset = InMemoryDataset(Torch_Dataset(
                data_path, train=False, download=True, transform=None
            ))
        '''

        #import ipdb 
        #ipdb.set_trace()

        scenario_train = ClassIncremental(train_dataset, increment=increment, 
                                        initial_increment= increment, transformations=[transforms.ToTensor()] )
        scenario_test = ClassIncremental(test_dataset, increment=increment, 
                                        initial_increment= increment, transformations=[transforms.ToTensor()] )        

    else: 
        output_dim = max(train_dataset.get_data()[1]) + 1
        
        scenario_train = ClassIncremental(train_dataset, increment=increment, 
                                        initial_increment= increment  + output_dim % increment)
        scenario_test = ClassIncremental(test_dataset, increment=increment, 
                                        initial_increment= increment  + output_dim % increment)
    
    
    return scenario_train, scenario_test, input_dim, output_dim


"""
train_datasets = = [SPLIT_Dataset(dataset_str, i,  self.split_path, transform=self.train_transform, train=True) for i in range(self.num_data_splits)] # assuming for now there are 5 splits for both datasets. 
test_datasets = [SPLIT_Dataset(dataset_str, i, self.split_path, transform=self.test_transform, train=False) for i in range(self.num_data_splits)]

if self.continual_learning and self.trainer.current_epoch % self.epochs_per_dataset == 0:
            if self.curr_index<(self.num_data_splits-1): # 5 datasets for now.
                # TODO: make this more general. 
                self.curr_index += 1
                print("SWITCHING DATASET BY INCREMENTING INDEX TO:", self.curr_index, "EPOCH IS:", self.trainer.current_epoch)

        if self.continual_learning: 
            self.train_data = self.train_datasets[self.curr_index]

if self.continual_learning:
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=True , num_workers=self.num_workers) for ds in self.test_datasets[:self.curr_index+1]]
        

"""

def give_torch_dataset(dataset_path, train):

    train_or_test = 'train' if train else "test"
    img_and_label_dir = dataset_path+ train_or_test+".pt"
    
    # loading all the images here as a large torch tensor rather than doing it one by one. 
    images, img_labels = torch.load(img_and_label_dir)

    if images.dtype is torch.uint8:
        #"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
        images = images.type(torch.float)/255

    #import ipdb 
    #ipdb.set_trace()

    return [images.numpy(), img_labels.numpy()]



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

def partial_permutation(size, perc):
    indices = np.arange(0, size)
    if perc == 0:
        return indices, indices
    rand_indices = np.random.choice(indices, int(size / 100.0 * perc))
    perm = np.random.permutation(rand_indices)
    indices[rand_indices] = perm
    return np.arange(0, size), indices

#perm_percentage is list [0, 10, 20, ...]
#number of tasks is len(perm_percentage)
def get_permuted_MNIST(perm_percentage):
    mnist_train = MNIST_TORCH('Data', train = True, download=True)
    mnist_test = MNIST_TORCH('Data', train = False, download=True)
    
    x_train_original = mnist_train.data
    y_train = mnist_train.targets
    x_test_original = mnist_test.data
    y_test = mnist_test.targets
    
    x_train_all = []
    y_train_all = []
    x_test_all = []
    y_test_all = []
    
    for id_, perm_perc in enumerate(perm_percentage):
        x_train_flat = torch.flatten(x_train_original, start_dim = 1)
        x_test_flat = torch.flatten(x_test_original, start_dim = 1)
        old, new = partial_permutation(784, perm_perc)
        x_train_flat[:, old] = x_train_flat[:, new]
        x_test_flat[:, old] = x_test_flat[:, new]
        x_train_all.append(x_train_flat.reshape(-1, 28, 28))
        x_test_all.append(x_test_flat.reshape(-1, 28, 28))
        y_train_all.append(y_train + 10 * id_)
        y_test_all.append(y_test + 10 * id_)
        
    train_x = np.concatenate(x_train_all)
    test_x = np.concatenate(x_test_all)
    train_y = np.concatenate(y_train_all)
    test_y =  np.concatenate(y_test_all)
    train_dataset = InMemoryDataset(train_x, train_y)
    test_dataset = InMemoryDataset(test_x, test_y)
    
    scenario_train = ClassIncremental(train_dataset, increment=[10] * len(perm_percentage))
    scenario_test = ClassIncremental(test_dataset, increment=[10] * len(perm_percentage))
    return scenario_train, scenario_test, 784, 10 * len(perm_percentage)
    
    
    
    