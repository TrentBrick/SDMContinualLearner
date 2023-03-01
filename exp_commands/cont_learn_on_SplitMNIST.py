import os
import sys
import copy
if 'exp_commands' not in os.getcwd():
    os.chdir('exp_commands/')
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('../py_scripts')
from py_scripts.model_params import *
from py_scripts.dataset_params import *
if 'exp_commands' in os.getcwd():
    os.chdir('..')
########### EXPERIMENTS TO RUN ##############

"""
SplitMNIST

Testing all models and random seeds directly on the embeddings. No pretrain.

Really need to Tune the GABA switch here so that it terminates and Top-K is fully implemented within the first task. 
"""

settings_for_all = dict(
    epochs_to_train_for = 2500, 
    classification=True,
    adversarial_attacks=False, 
    dataset = DataSet.SPLIT_MNIST,
    epochs_per_dataset = 500,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,

    num_binary_activations_for_gaba_switch=5000000,
    k_transition_epochs = 200,

    opt="SGD",
    lr=0.05,
    
    ewc_memory_beta=0.005,
    ewc_memory_importance=200,
    mas_importance=0.5,
    l2_importance = 10,
    si_beta=0.005,
    si_importance=1500,
)

name_suffix = "_MNIST_NoPretrain" 


exp_list = [

    #NEED TO PUT IN THE LOAD PATHS

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_10K", 
        nneurons=[10000], 
        k_approach="GABA_SWITCH_ACT_BIN", 
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC_10K",
        nneurons=[10000],
        cl_baseline = "EWC-MEMORY",
    ),

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=1", 
        k_min=1, 
        k_approach="GABA_SWITCH_ACT_BIN", 
    ),
    
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC",
        cl_baseline = "EWC-MEMORY",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC",
        cl_baseline = "EWC-MEMORY",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "MAS",
        cl_baseline = "MAS",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "SI",
        cl_baseline = "SI",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
    ),

    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopK",
    ),

    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)