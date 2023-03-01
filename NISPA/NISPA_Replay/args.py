import argparse


def get_args():
    
    #Experiment Setting
    parser = argparse.ArgumentParser(description='CL Experiment')
    parser.add_argument('--experiment_name', type=str, default = "Test")
    parser.add_argument('--experiment_note', type=str, default = "")
    parser.add_argument('--dataset', type=str, default = "emnist_fmnist", choices=["mnist",  "fmnist",
                                                                              'emnist', 'emnist_fmnist', 'cifar100', "cifar10"])
    
    #Architectural Settings
    parser.add_argument('--model', type=str, default = "mlp", choices=["conv", "mlp"])
    parser.add_argument('--seed', type=int,  default=0)
    
    #Pruning Settings
    parser.add_argument('--prune_perc', type=float, default=90)
    
    #Continual Learning Settings
    parser.add_argument('--class_per_task', type=int, default=2)
    
    #Optimization Settings
    parser.add_argument('--optimizer', type=str, default = "Adam", choices=["adam", "ada_delta" ,"SGD"])
    parser.add_argument('--learning_rate', type=float, default = 0.05)
    parser.add_argument('--batch_size', type=int, default = 256)
    
    #Algorithm
    parser.add_argument('--recovery_perc', type=float, default = 0.2) #a_f hyperparameter
    parser.add_argument('--phase_epochs', type=int, default = 5) #e hyperparameter
    parser.add_argument('--reinit',  type=int, default = 0)     #Reinitialize connections that are not frozed
    parser.add_argument('--grow', type=int, default = 1)        #Grow connections 
    parser.add_argument('--p_step_size', type=str, default = "cosine", choices=['cosine', 'exp_decay', "linear"]) # how to determine p
    parser.add_argument('--step_size_param',  type=float, default = 30) #k hyperparameter
    parser.add_argument('--grow_init', type=str, default = "normal", choices=['normal', 'zero']) #initilization of new connections
    parser.add_argument('--rewire_algo', type=str, default = "full_random", choices=['full_random']) 
     

    #Replay Mechanism
    #How many samples we are saving for replay
    parser.add_argument('--memo_size', type=int, default=1000)
    #How many times replayed per epoch
    parser.add_argument('--replay_lambda', type=float, default=1)

    #min phases
    parser.add_argument('--min_phases', type=int, default=5) 
    
    #Parse commandline arguments
    return parser.parse_args()