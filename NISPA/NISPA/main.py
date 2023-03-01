import argparse
from dataset import get_dataset
from NISPA import learner
import sys
import os


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('Data'):
        os.makedirs('Data')

    #Experiment Setting
    parser = argparse.ArgumentParser(description='CL Experiment')
    parser.add_argument('--experiment_name', type=str, default = "Test")
    parser.add_argument('--experiment_note', type=str, default = "")
    parser.add_argument('--dataset', type=str, default = "mnist", choices=["mnist", "pmnist_partial", "fmnist", 'emnist', 'emnist_fmnist', 'cifar100', "cifar10", "split_cifar100", "split_cifar10", "split_mnist"])
    
    #Only used for pmnist_partial
    parser.add_argument('--perm_perc', nargs='+', type=float,  default = [0, 10, 100] )
    
    #Architectural Settings
    parser.add_argument('--model', type=str, default = "mlp", choices=["mlp", "conv"])
    parser.add_argument('--seed', type=int,  default=0)
    
    #Pruning Settings
    parser.add_argument('--prune_perc', type=float, default=90)
    
    #Continual Learning Settings
    parser.add_argument('--class_per_task', type=int, default=2) 
    
    #Optimization Settings
    parser.add_argument('--optimizer', type=str, default = "adam", choices=["adam", "ada_delta" ,"SGD"])
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    parser.add_argument('--batch_size', type=int, default = 512)
    
    #Algorithm
    parser.add_argument('--recovery_perc', type=float, default = 0.75) #a_f hyperparameter
    parser.add_argument('--phase_epochs', type=int, default = 2) #e hyperparameter
    parser.add_argument('--reinit',  type=int, default = 1)     #Reinitialize connections that are not frozed
    parser.add_argument('--grow', type=int, default = 1)        #Grow connections 
    parser.add_argument('--p_step_size', type=str, default = "cosine", choices=['cosine', 'exp_decay','linear']) # how to determine p
    parser.add_argument('--step_size_param',  type=float, default = 30) #k hyperparameter
    parser.add_argument('--grow_init', type=str, default = "normal", choices=['uniform','normal', 'zero']) #initilization of new connections
    #full_random type1 + type 2 connections | flow_random type 2 connections | forwardT_random type 2 connections 
    parser.add_argument('--rewire_algo', type=str, default = "full_random", choices=['full_random', 'forwardT_random', 'flow_random']) 
     
    #Seperate Heads
    #Set both to one for multihead setting
    parser.add_argument('--multihead', type=int, default = 1)
    parser.add_argument('--mask_outputs', type=int, default = 1)
    
    #Parse commandline arguments
    args = parser.parse_args()
    scenario_train, scenario_test, input_dim, output_dim = get_dataset(args.dataset, increment = args.class_per_task, args= args)
    
    accs_after_finetune = []
    freeze_percs = []
    optimal_ps = []
    task_ps = []
    num_frozen_units = []
    
    learner = learner.Learner(args, input_dim, output_dim, scenario_train, scenario_test, deterministic = True)
    
    for task_id, taskset in enumerate(scenario_train):
       optimal_p,  percentange_of_frozen_conns, acc_after_fine_tuning, _, list_of_ps, num_important_units =  learner.task_train(task_id, taskset, None)
       task_ps =  task_ps + list_of_ps
       num_frozen_units.append(num_important_units)
       accs_after_finetune.append(acc_after_fine_tuning)
       freeze_percs.append(percentange_of_frozen_conns)
       optimal_ps.append(optimal_p)
       
    #Write results   
    import xlsxwriter
    row = 0
    column = 0
    workbook = xlsxwriter.Workbook('./results/{}.xlsx'.format(args.experiment_name))
    worksheet = workbook.add_worksheet()
    setting =  ' '.join(sys.argv[1:])
    worksheet.write(row, column, "Results: {}".format(setting))
    #Write selected ps
    column += 1
    row = 0
    worksheet.write(row, column, "Final P")
    for i in range(len(optimal_ps)):
        row += 1
        worksheet.write(row, column, optimal_ps[i])
        
    #Write After accs
    column += 1
    row = 0
    worksheet.write(row, column, "Test Accuracy")
    for i in range(len(accs_after_finetune)):
        row += 1
        worksheet.write(row, column, accs_after_finetune[i])
        
    workbook.close()