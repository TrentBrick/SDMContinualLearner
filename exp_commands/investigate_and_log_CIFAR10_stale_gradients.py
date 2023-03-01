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
Logging the number of dead neurons created by the different optimizers and the gradients applied to each of them. Uses CIFAR10 pixels as the manifold is more complex and more prone to result in dead neurons that can be analyzed. 
"""

settings_for_all = dict(
    epochs_to_train_for = 1000, 
    dataset = DataSet.CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    num_binary_activations_for_gaba_switch=10000000,

    count_dead_train_neurons = True, 
    log_gradients=True,
    start_epoch_log_grads =95,
)

# LogGrads
name_suffix = "_LogGrads_TrackTrainandVal_10MSwitch_RawCIFAR_StaleGradients_Longer" 

exp_list = [
    
    #####
    
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_RMSProp_lr=0.0005",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="RMSProp",
        lr=0.0005,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGDM_lr=0.09",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGDM",
        lr=0.09,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_SGD_lr=0.2",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGD",
        lr=0.2,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Adam_lr=0.001",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="Adam",
        lr=0.001,
    ),
    
    
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)