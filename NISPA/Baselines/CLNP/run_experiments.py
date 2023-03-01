from continuum import ClassIncremental
from continuum.datasets import MNIST, FashionMNIST, EMNIST, Fellowship, CIFAR100, CIFAR10
import argparse
import numpy as np
import clnp_learner
import random
import torch
import xlsxwriter
from collections import OrderedDict
import sys
import os


def list2acc_table(accuracies, num_tasks):
    acc_table = OrderedDict()
    #Build dict
    for i in range(num_tasks):
        acc_table[i+1] = {}
        for j in range(num_tasks):
            acc_table[i+1][j+1] = 0
    
    #Fill dict
    for time, task_acc in enumerate(task_accuracies):
        for back_id, back_acc in task_acc:
            acc_table[back_id+1][time+1] = back_acc
    return acc_table
    
def export_results(filename, setting, acc_table):
    #Export Accuracy Table to Excel
    row = 0
    column = 0
    workbook = xlsxwriter.Workbook('./results/{}.xlsx'.format(filename))
    worksheet = workbook.add_worksheet()
    worksheet.write(row, column, "Results: {}".format(setting))
    for i, task in enumerate(acc_table.keys(), start = 1):
        worksheet.write(i, 0, "Task {} Acc".format(task))
        worksheet.write(0, i, "End of Task {}".format(task))
    tasks = list(acc_table.keys())
    for time_point in tasks:
        for acc_task in tasks[:int(time_point)]:
            worksheet.write(int(acc_task),  int(time_point),str(acc_table[acc_task][time_point]))
    workbook.close()
    
    

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

    if dataset == "cifar10":
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
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('Data'):
        os.makedirs('Data')
        
    #Experiment Setting
    parser = argparse.ArgumentParser(description='CL Experiment')
    parser.add_argument('--experiment_name', type=str, default = "Zero")
    parser.add_argument('--dataset', type=str, default = "mnist", choices=["mnist",  "fmnist", 'emnist', "emnist_fmnist",  'cifar100', 'cifar10'])
    
    #Architectural Settings
    parser.add_argument('--model', type=str, default = "mlp", choices=["mlp", "conv"])
    parser.add_argument('--seed', type=int, default=0)
    
    #Continual Learning Settings
    parser.add_argument('--class_per_task', type=int, default=5) 
    
    #Optimization Settings
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--l1_alpha', type=float, default = 10e-5)
    parser.add_argument('--epochs', type=int, default = 10)
    
    #Algorithm
    parser.add_argument('--m_perc', type=float, default = 0.2)
    parser.add_argument('--theta_step', type=float, default = 0.1)
    
    #Parse commandline arguments
    args = parser.parse_args()
    setting =  ' '.join(sys.argv[1:])
    
    #Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
            
            
    scenario_train, scenario_test, input_dim, output_dim = get_dataset(args.dataset, increment = args.class_per_task)
    if args.model == 'conv':
        learner = clnp_learner.CL_Learner_Conv(args, input_dim, output_dim, scenario_train, scenario_test)
    else:
        learner = clnp_learner.CL_Learner_MLP(args, input_dim, output_dim, scenario_train, scenario_test)
    task_accuracies = []
    for task_id, taskset in enumerate(scenario_train):
        accs = learner.task_train(task_id, taskset)
        print(accs)
        task_accuracies.append(accs)
        
    acc_table = list2acc_table(task_accuracies, len(scenario_train))
    export_results(args.experiment_name, setting, acc_table)
        