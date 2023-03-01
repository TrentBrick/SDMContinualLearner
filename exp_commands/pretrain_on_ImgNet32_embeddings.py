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
Pretraining all models on ImageNet32 Convmixer embeddings.
"""

settings_for_all = dict(
    epochs_to_train_for = 300, 
    num_binary_activations_for_gaba_switch=500000,
    dataset = DataSet.Cached_ConvMixer_ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    epochs_per_dataset = 500,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    
    #count_dead_train_neurons=True, 
    #investigate_cont_learning = True,
    #investigate_cont_learning_log_every_n_epochs = 10, 
)

name_suffix = "_PosWeights_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32" # can set to None. 

exp_list = [
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM",
        k_approach = "GABA_SWITCH_ACT_BIN", 
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_10K",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        nneurons=[10000],
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecayMask",
        k_approach = "LINEAR_DECAY_MASK", 
        k_transition_epochs=100,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_LinearDecaySub",
        k_approach = "LINEAR_DECAY_SUBTRACT", 
        k_transition_epochs=100,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_FlatMask",
        k_approach = "FLAT_MASK", 
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_k=1",
        k_approach = "GABA_SWITCH_ACT_BIN", 
        num_binary_activations_for_gaba_switch=500000,
        k_min=1,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Bias",
        use_bias=True,
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_NoL2Norms",
        norm_addresses=False,
    ),
    

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
        #lr=0.01,
        opt="SGD",
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGDM",
        #lr=0.01,
        opt="SGDM",
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_Adam",
        lr=0.001,
        opt="Adam",
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopKDefault",
        opt="SGD",
        #lr=0.03,
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopK_SDM_Control",
        use_bias=False,
        norm_addresses=True,  # L2 norm of neuron addresses and inputs.
        k_approach = "GABA_SWITCH_ACT_BIN", 
        opt="SGDM",
        lr=0.03,
    ),


    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD_10K",
        #lr=0.01,
        opt="SGD",
        nneurons=[10000],
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGDM_10K",
        #lr=0.01,
        opt="SGDM",
        nneurons=[10000],
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_Adam_10K",
        lr=0.001,
        opt="Adam",
        nneurons=[10000],
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopKDefault_10K",
        opt="SGD",
        nneurons=[10000],
        #lr=0.03,
    ),




]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)

