import os
import sys
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
Getting nice receptive fields for the 10K neuron model. And for ReLU Baselines as a comparison. 
"""

settings_for_all = dict(
    epochs_to_train_for = 1000, 
    dataset = DataSet.CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = True,
    num_binary_activations_for_gaba_switch=20000000,
    nneurons=[10000],
    count_dead_train_neurons = True, 
)

name_suffix = "_NiceReceptiveFields_10MSwitch_RawCIFAR_StaleGradients_Longer" 


exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGD_lr=0.2_PosWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGD",
        lr=0.1,
        all_positive_weights = True, 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGDM_lr=0.09_PosWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGDM",
        lr=0.03,
        all_positive_weights = True, 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGD_lr=0.2_NegWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGD",
        lr=0.1,
        all_positive_weights = False, 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGDM_lr=0.03_PosWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGDM",
        lr=0.03,
        all_positive_weights = True, 
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGDM_lr=0.03_NegWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGDM",
        lr=0.03,
        all_positive_weights = False, 
    ),
    
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "FFN_SGDM_lr=0.03_NegWeights",
        opt="SGDM",
        lr=0.03,
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "FFN_Adam_lr=0.001_NegWeights",
        opt="Adam",
        lr=0.001,
    ),
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)