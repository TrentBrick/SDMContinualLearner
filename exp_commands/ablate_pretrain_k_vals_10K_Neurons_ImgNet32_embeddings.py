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
10K neurons. Pretraining with different K values. Full 300 epochs of training. And double the bare min for the GABA switch. Can compare this to the full model which had a much larger GABA switch if 
        desired. 
"""

settings_for_all = dict(
    epochs_to_train_for = 300, 
    dataset = DataSet.Cached_ConvMixer_ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    nneurons=[10000],
)

name_suffix = "_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10" 

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