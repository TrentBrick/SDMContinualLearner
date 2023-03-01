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
Testing continual learning with loads of different k values for SDM. 
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

name_suffix = "_KVals_Ablation_TEST_ConvMixer_WTransforms_ImageNet32_ImageNet32" 

exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=1",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=1,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=1_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=5",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=5,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=5_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=10",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=10,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=10_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),


    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=3",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=3,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=3_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=8",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=8,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=8_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=12",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=12,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=12_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=15",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=15,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=15_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=18",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=18,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=18_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=20",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=20,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=20_KValAblation_2_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),


    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=25",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=25,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=25_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=50",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=50,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=50_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=100",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=100,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=100_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=150",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=150,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=150_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=250",
        all_positive_weights=False,
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=250,
        load_path="experiments/TopK_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/SDM_k=250_KValAblation_Pretrains_ConvMixer_WTransforms_ImageNet32_CIFAR10",
    ),

    # during the test stage PUT IN DIFFERENT LEARNING RATES FOR THE DEFAULT MODELS!!!
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)