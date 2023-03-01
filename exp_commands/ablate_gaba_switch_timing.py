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
Pretraining with different GABA switch values. Using even smaller values now!!
"""

settings_for_all = dict(
    epochs_to_train_for = 30,
    dataset = DataSet.Cached_ConvMixer_ImageNet32,
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
)

name_suffix = "_GABASwitchAblation_Pretrains_ConvMixer_ImageNet32" 

exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_1x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=10000000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.5x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=5000000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.25x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=2500000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.1x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=1000000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.05x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.025x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=250000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.01x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=100000,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.005x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=50000,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.0025x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=25000,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.001x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=10000,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_0.0005x",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=5000,
    ),
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)