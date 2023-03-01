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
All seeds and the best models plus all baselines. SuperLong Cifar10 testing. All pos weights models not accounted for in the negative weight runs. 
"""

settings_for_all = dict(
    epochs_to_train_for = 10000, 
    classification=True,
    adversarial_attacks=False, 
    epochs_per_dataset = 2000,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    opt="SGD",
    lr=0.05,
)

name_suffix = "_SGD_SuperLong_TEST_ConvMixer_Embedding" 


og_exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_10K",
        nneurons=[10000],
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        k_approach = "GABA_SWITCH_ACT_BIN", 
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
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecaySub",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        k_transition_epochs=100,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_LinearDecaySub_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_FlatMask",
        k_approach = "FLAT_MASK", 
        k_transition_epochs=100,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_FlatMask_FixKMaskValues_PosWs_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        opt="SGD",
        lr=0.05,
    ),

    
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecayMask",
        k_approach = "LINEAR_DECAY_MASK", 
        k_transition_epochs=100,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_LinearDecayMask_FixKMaskValues_PosWs_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        opt="SGD",
        lr=0.05,
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NegWeight",
        all_positive_weights=False,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
        k_approach = "GABA_SWITCH_ACT_BIN",
        opt="SGD",
        lr=0.05,
    ),


    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC_SGD_10K",
        cl_baseline = "EWC-MEMORY",
        nneurons=[10000],
        lr=0.05,
        opt="SGD",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGDM_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC_10K",
        cl_baseline = "EWC-MEMORY",
        nneurons=[10000],
        opt="SGDM",
        lr=0.03,
       
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC",
        cl_baseline = "EWC-MEMORY",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        opt="SGDM",
        lr=0.03,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC_SGD",
        lr=0.05,
        opt="SGD",
        cl_baseline = "EWC-MEMORY",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),


    # MAS
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "MAS_SGD_10K",
        nneurons=[10000],
        lr=0.05,
        opt="SGD",
        cl_baseline = "MAS",
        mas_importance=10,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGDM_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_MAS_10K",
        nneurons=[10000],
        opt="SGDM",
        lr=0.03,
       
        cl_baseline = "MAS",
        mas_importance=10,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_MAS",
        cl_baseline = "MAS",
        mas_importance=10,
        opt="SGDM",
        lr=0.03,
        load_path="experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/SDM_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "MAS_SGD",
        lr=0.05,
        opt="SGD",
        cl_baseline = "MAS",
        mas_importance=10,
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "L2_SGD",
        lr=0.05,
        opt="SGD",
        cl_baseline = "L2",
        l2_importance=30, 
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Dropout_SGD",
        lr=0.05,
        opt="SGD",
        dropout_prob=0.03, 
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "SI_SGD_10K",
        nneurons=[10000],
        lr=0.05,
        opt="SGD",
        cl_baseline = "SI",
        si_importance=1350, 
        si_beta = 0.02,
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGDM_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "SI_SGD",
        lr=0.05,
        opt="SGD",
        cl_baseline = "SI",
        si_importance=1350, 
        si_beta = 0.02,
        load_path="experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),

]

exp_list = []
for rand_seed in [None, 3, 15, 27, 97]:

    for e in og_exp_list:

        temp_exp = copy.deepcopy(e)

        if rand_seed is not None: 
            temp_exp['dataset'] = DataSet[DataSet.SPLIT_Cached_ConvMixer_CIFAR10.name +f"_RandSeed_{rand_seed}"]
        else: 
            temp_exp['dataset'] = DataSet.SPLIT_Cached_ConvMixer_CIFAR10

        temp_exp['test_name'] = temp_exp['test_name']+f"_RS_{rand_seed}"

        exp_list.append(temp_exp)

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)