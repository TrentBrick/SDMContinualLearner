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
Testing dead neuron numbers for gaba switch vs linear subtraction 
"""

settings_for_all = dict(
    epochs_to_train_for = 25, 
    dataset = DataSet.Cached_ConvMixer_ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
)

name_suffix = "_QuickDeadNeurons_Pretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32" 

exp_list = [
    
    #####
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_500K_PosWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearSubtract_KTrans1_PosWeights",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        k_transition_epochs=1,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearSubtract_KTrans3_PosWeights",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        k_transition_epochs=3,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_500K_NegWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        all_positive_weights=False,
        num_binary_activations_for_gaba_switch=500000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearSubtract_KTrans1_NegWeights",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        all_positive_weights=False,
        k_transition_epochs=1,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearSubtract_KTrans3_NegWeights",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        all_positive_weights=False,
        k_transition_epochs=3,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NegWeights",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        all_positive_weights=False,
        num_binary_activations_for_gaba_switch=250000,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearSubtract_KTrans5_NegWeights",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        all_positive_weights=False,
        k_transition_epochs=5,
    ),

    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)