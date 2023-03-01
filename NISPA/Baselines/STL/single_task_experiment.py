from continuum import ClassIncremental
from continuum.datasets import MNIST, FashionMNIST, EMNIST, Fellowship, CIFAR100, CIFAR10
import argparse
import single_task_learner
import os

def get_dataset(dataset, increment):
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
        train_dataset = Fellowship([EMNIST('Data', train = True, download=True, split='balanced'), FashionMNIST('Data', train = True, download=True)])
        test_dataset = Fellowship([EMNIST('Data', train = False, download=True, split='balanced'), FashionMNIST('Data', train = False, download=True)])
        scenario_train = ClassIncremental(train_dataset, increment=[10, 13, 13, 11, 10])
        scenario_test = ClassIncremental(test_dataset, increment=[10, 13, 13, 11, 10])
        output_dim = max(train_dataset.get_data()[1]) + 1
        input_dim  = 28*28
        return scenario_train,  scenario_test, input_dim, output_dim
        
    output_dim = max(train_dataset.get_data()[1]) + 1
    
    scenario_train = ClassIncremental(train_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment)
    scenario_test = ClassIncremental(test_dataset, increment=increment, 
                                      initial_increment= increment  + output_dim % increment)
    return scenario_train, scenario_test, input_dim, output_dim


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('Data'):
        os.makedirs('Data')
    #Experiment Setting
    parser = argparse.ArgumentParser(description='CL Experiment')
    parser.add_argument('--experiment_name', type=str, default = "SingleTask")
    parser.add_argument('--experiment_note', type=str, default = "")
    parser.add_argument('--dataset', type=str, default = "mnist", choices=["mnist", "fmnist", 'emnist', "emnist_fmnist", "cifar100", "cifar10"])
    
    #Architectural Settings
    parser.add_argument('--model', type=str, default = "mlp", choices=["mlp", "conv"])
    parser.add_argument('--activation', type=str, default = "relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop_out', type=int, default = 0)
    
    #Pruning Settings
    parser.add_argument('--prune_perc', type=float, default=90)
    
    #Continual Learning Settings
    parser.add_argument('--class_per_task', type=int, default=2) 
    
    #Optimization Settings
    parser.add_argument('--optimizer', type=str, default = "adam", choices=["adam", "ada_delta" ,"SGD"])
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    parser.add_argument('--weight_decay', type=float, default = 0.0)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_norm', type=int, default = 0)
    parser.add_argument('--batch_size', type=int, default = 128)
    
    #Seperate Heads
    parser.add_argument('--mask_outputs', type=int, default = 1)
    parser.add_argument('--multihead', type=int, default = 1)
    
    #Parse commandline arguments
    args = parser.parse_args()   
   
    scenario_train, scenario_test, input_dim, output_dim = get_dataset(args.dataset, increment = args.class_per_task)
    model = single_task_learner.Single_Task_Learner(args, input_dim, output_dim, scenario_train, scenario_test, deterministic = True)
    model.train_all()