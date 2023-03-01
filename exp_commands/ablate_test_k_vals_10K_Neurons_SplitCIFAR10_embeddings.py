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
10K neurons. TESTING with different K values. Full 300 epochs of training. And double the bare min for the GABA switch. Can compare this to the full model which had a much larger GABA switch if 
        desired. 
"""

settings_for_all = dict(
    epochs_to_train_for = 2500, 
    epochs_per_dataset = 500,
    dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10,
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    nneurons=[10000],
)

name_suffix = "_10K_KValAblation_TEST_ConvMixer_WTransforms_ImageNet32_CIFAR10" 

exp_list = [ 
   dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=1", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=1, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=1_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=5", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=5, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=5_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=150", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=150, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=150_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=200", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=200, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=200_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=250", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=250, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=250_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=50", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=50, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=50_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=25", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=25, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=25_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict( 
        
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=10", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=10, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=10_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
   dict(  
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=100", 
        all_positive_weights=False, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000, 
        k_min=100, 
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=100_equalTrains_10K_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10", 
    ), 
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)