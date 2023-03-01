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
Different SDM optimizers and their effects on the dead neuron problem. 
"""

settings_for_all = dict(
    epochs_to_train_for = 2500, 
    dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    epochs_per_dataset = 500,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
)

name_suffix = "_Optimizer_Ablation_TEST_ConvMixer_Embeddings" 

exp_list = [

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_SGD", 
        opt="SGD", 
        lr=0.08, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ), 

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_SGDM", 
        opt="SGDM", 
        lr=0.03, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ), 

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_RMSProp", 
        opt="RMSProp", 
        lr=0.0005, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_RMSProp_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32", 
    ), 
   dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_Adam", 
        opt="Adam", 
        lr=0.001, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_Adam_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32", 
    ),
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)