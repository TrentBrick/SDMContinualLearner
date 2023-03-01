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
SplitMNIST

Testing models with and without output layer bias terms. 
"""

settings_for_all = dict(
    epochs_to_train_for = 2500, 
    classification=True,
    adversarial_attacks=False, 
    dataset = DataSet.SPLIT_MNIST,
    epochs_per_dataset = 500,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,

    num_binary_activations_for_gaba_switch=5000000,
    k_transition_epochs = 200,

    opt="SGD",
    lr=0.05,
    
    ewc_memory_beta=0.005,
    ewc_memory_importance=200,
    mas_importance=0.5,
    l2_importance = 10,
    si_beta=0.005,
    si_importance=1500,
)

name_suffix = "_MNIST_NoPretrain_Ablate_BiasTerms" 


exp_list = [

    #NEED TO PUT IN THE LOAD PATHS

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=1_yesOutBias", 
        k_min=1, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        use_output_layer_bias=True
    ),

    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=1_noOutBias_yesGranBias", 
        k_min=1, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        use_bias = True,
        use_output_layer_bias=False,
    ),
    
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC_noOutBias",
        cl_baseline = "EWC-MEMORY",
        use_output_layer_bias=False,
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "MAS_noOutBias",
        cl_baseline = "MAS",
        use_output_layer_bias=False,
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "SI_noOutBias",
        cl_baseline = "SI",
        use_output_layer_bias=False,
    ),

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD_noOutBias",
        use_output_layer_bias=False,
    ),

    dict(
        model_style= ModelStyles.FFN_TOP_K,
        test_name= "TopK_noOutBias",
        use_output_layer_bias=False,
    ),

    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)