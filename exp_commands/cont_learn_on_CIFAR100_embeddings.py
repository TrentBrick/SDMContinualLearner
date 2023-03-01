import os
import sys
import copy
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
Testing CIFAR100. Using ImageNet32 pretrain. Using ConvMixer embeddings. 
"""

settings_for_all = dict(
    epochs_to_train_for = 25000, 
    classification=True,
    adversarial_attacks=False, 
    dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR100,
    epochs_per_dataset = 500,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,

    opt="SGD",
    lr=0.05,

    ewc_memory_beta=0.08,
    ewc_memory_importance=100,
)

name_suffix = "_TEST_CIFAR100" 

exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_10K",
        
        nneurons=[10000],
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),
    

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC_10K",
        cl_baseline = "EWC-MEMORY",
        nneurons=[10000],

        
 
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGDM_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC_10K",
        cl_baseline = "EWC-MEMORY",
        nneurons=[10000],

        opt="SGDM",
        lr=0.03,
       
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC",
        cl_baseline = "EWC-MEMORY",
        opt="SGDM",
        lr=0.03,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC",
        cl_baseline = "EWC-MEMORY",
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "MAS",
        cl_baseline = "MAS",
        mas_importance=10,
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),


    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        k_approach = "GABA_SWITCH_ACT_BIN",  
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=1",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=1,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_k=1_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopK",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/TopKDefault_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32", 
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGD_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "SI",
        cl_baseline = "SI",
        si_importance=1350, 
        si_beta = 0.02,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGDM_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    

    
    
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)