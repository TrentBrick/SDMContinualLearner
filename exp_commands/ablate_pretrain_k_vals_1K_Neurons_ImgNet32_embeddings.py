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
Pretraining with different K values. Full 300 epochs of training. And double the bare minimum for the GABA switch.
"""

settings_for_all = dict(
    epochs_to_train_for = 300, 
    dataset = DataSet.Cached_ConvMixer_ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
)

name_suffix = "_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10" 

exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=1",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=1,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=5",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=5,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=3",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=3,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=8",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=8,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=12",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=12,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=15",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=15,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=18",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=18,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=20",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=20,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=23",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=23,
    ),


    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=10",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=10,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=25",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=25,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=50",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=50,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=100",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=100,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=150",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=150,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=200",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=200,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=250",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=250,
    ),
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)