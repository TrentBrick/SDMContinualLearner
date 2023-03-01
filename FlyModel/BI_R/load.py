import copy
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.manipulate import ReducedDataset, SubDataset, TransformedDataset, permutate_image_pixels
from torch import tensor, reshape
from torch.utils.data import Dataset, DataLoader, TensorDataset

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None, valid_prop=0.):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name in ('mnist28') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # if relevant, select "train" or "validation"-set from training-part of data
    # NOTE: this split assumes order of items in training-dataset is random!
    # (e.g., not first all samples from clas 1, then all samples from class 2, etc.)
    if (type=='train' or type=='valid') and valid_prop>0:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_prop * dataset_size))
        if type=='train':
            indices_to_use = indices[split:]
        elif type=='valid':
            indices_to_use = indices[:split]
        dataset = ReducedDataset(dataset, indices_to_use)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


##-------------------------------------------------------------------------------------------------------------------##


def get_singletask_experiment(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-task experiment.'''

    # Define data-type
    if name == "MNIST":
        data_type = 'mnist'
    elif name == "MNIST28":
        data_type = 'mnist28'
    elif name == "CIFAR10":
        data_type = 'cifar10'
    elif name == "CIFAR100":
        data_type = 'cifar100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[data_type]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[data_type+"_denorm"]
    trainset = get_dataset(data_type, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(data_type, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config


def get_multitask_experiment(name, scenario, tasks, data_dir="./store/datasets", normalize=False, augment=False,
                             only_config=False, verbose=False, exception=False, only_test=False):
    '''Load, organize and return train- and test-dataset for requested multi-task experiment.'''

    ## NOTE: option 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            if not only_test:
                train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                            target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # specify transformed datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class', 'all') else None
                if not only_test:
                    train_datasets.append(TransformedDataset(
                        train_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                        target_transform=target_transform
                    ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment '{}' cannot have more than 10 tasks!".format(name))
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))

    elif name == 'combined':
        # check for number of tasks
        if tasks>20:
            raise ValueError("Experiment '{}' cannot have more than 20 tasks!".format(name))
        # configurations
        config = DATASET_CONFIGS['combined']
        classes_per_task = int(np.floor(20 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(20))) if exception else np.random.permutation(list(range(20)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                train1 = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
                train2 = get_dataset('fashionmnist', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
                train2.targets = train2.targets + 10
                mnist_train = ConcatDataset([train1, train2])

                test1 = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
                test2 = get_dataset('fashionmnist', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
                test2.targets = test2.targets + 10
                mnist_test = ConcatDataset([test1, test2])

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))

    elif name == 'CIFAR100':
        # check for number of tasks
        if tasks>100:
            raise ValueError("Experiment 'CIFAR100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))

    elif name == 'mnist_combined':
        datadir = '/home/yang/Documents/Research/Lifelong_learning/notebooks/processed_data/'
        # check for number of tasks
        if tasks>20:
            raise ValueError("Experiment 'mnist_combined' cannot have more than 20 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist_combined']
        classes_per_task = int(np.floor(20 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            #permutation = np.random.permutation(list(range(20)))
            #target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                combined_train = pd.read_csv(datadir + 'fmnist_mnist_combined_train.csv', header=None)
                train_labels = pd.read_csv(datadir + 'fmnist_mnist_combined_train_label.csv', header=None)
                train_labels = np.array([x[0] for x in np.array(train_labels)])
                combined_train, train_labels = map(tensor, (combined_train.T.to_numpy(), train_labels))
                combined_train = reshape(combined_train,(combined_train.shape[0],10,10))
                train_com = TensorDataset(combined_train.float(), train_labels)

            combined_test = pd.read_csv(datadir + 'fmnist_mnist_combined_test.csv', header=None)
            test_labels = pd.read_csv(datadir + 'fmnist_mnist_combined_test_label.csv', header=None)
            test_labels = np.array([x[0] for x in np.array(test_labels)])
            combined_test, test_labels = map(tensor, (combined_test.T.to_numpy(), test_labels))
            combined_test = reshape(combined_test,(combined_test.shape[0],10,10))
            test_com = TensorDataset(combined_test.float(), test_labels)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(train_com, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(test_com, labels, target_transform=target_transform))
    elif name == 'precifar100':
        datadir = '/home/yang/Documents/Research/Lifelong_learning/notebooks/processed_data/'
        # check for number of tasks
        if tasks>100:
            raise ValueError("Experiment 'precifar100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['precifar100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            #permutation = np.random.permutation(list(range(100)))
            #target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                combined_train = pd.read_csv(datadir + 'cifar100_train_resnet18_padded.csv', header=None)
                train_labels = pd.read_csv(datadir + 'cifar100_train_label_resnet18.csv', header=None)
                train_labels = np.array([x[0] for x in np.array(train_labels)])
                combined_train, train_labels = map(tensor, (combined_train.to_numpy(), train_labels))
                combined_train = reshape(combined_train,(combined_train.shape[0],23,23))
                train_com = TensorDataset(combined_train.float(), train_labels)

            combined_test = pd.read_csv(datadir + 'cifar100_test_resnet18_padded.csv', header=None)
            test_labels = pd.read_csv(datadir + 'cifar100_test_label_resnet18.csv', header=None)
            test_labels = np.array([x[0] for x in np.array(test_labels)])
            combined_test, test_labels = map(tensor, (combined_test.to_numpy(), test_labels))
            combined_test = reshape(combined_test,(combined_test.shape[0],23,23))
            test_com = TensorDataset(combined_test.float(), test_labels)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(train_com, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(test_com, labels, target_transform=target_transform))

    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["cifar100_denorm"]

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)