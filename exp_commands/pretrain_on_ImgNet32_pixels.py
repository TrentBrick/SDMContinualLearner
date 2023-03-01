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
Pretraining all of the main models on raw imagenet.
"""

settings_for_all = dict(
    epochs_to_train_for = 100, 
    dataset = DataSet.ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    num_binary_activations_for_gaba_switch=10000000,
    k_transition_epochs = 50,
)

name_suffix = "_LargerGABA_Pretrains_Raw_ImageNet32" 

exp_list = [
    
    # ADD IN EWC AFTER TRAINING!!!!

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
        opt="SGD",
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGDM",
        opt="SGDM",
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopKDefault",
        opt="SGD",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecayMask",
        k_approach = "LINEAR_DECAY_MASK", 
        
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecaySub",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_FlatMask",
        k_approach = "FLAT_MASK", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_10K",
        nneurons=[10000],
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM",
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=1",
        k_min=1,
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=25",
        k_min=25,
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=100",
        k_min=100,
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Bias",
        use_bias=True,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NoL2Norms",
        norm_addresses=False,
    ),
    
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NegWeights",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),

    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)