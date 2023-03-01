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
Training default SDM model vs ReLU vs SDM linear decay vs TopK in the continual learning setting to see exactly how it avoids
    catastrophic forgetting by logging lots of things!
"""

settings_for_all = dict(
    epochs_to_train_for = 1500, 
    dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    epochs_per_dataset = 300,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    investigate_cont_learning = True,
    investigate_cont_learning_log_every_n_epochs = 100, 
    checkpoint_every_n_epochs = 100
)

name_suffix = "_Investigate_ContLearning" # can set to None. 

exp_list = [
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM",
        all_positive_weights=False,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NoL2Norms",
        norm_addresses=False,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_NoL2Norms_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        all_positive_weights=False,
       
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecayMask",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_LinearDecayMask_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        all_positive_weights=False,
        k_approach = "LINEAR_DECAY_MASK", 
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopKDefault",
        opt="SGD",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/TopKDefault_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32", 
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
        opt="SGD",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGD_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),
    
] 

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)