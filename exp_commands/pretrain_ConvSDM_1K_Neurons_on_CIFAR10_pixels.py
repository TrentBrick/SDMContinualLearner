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
1K neurons. 
Much longer GABA switch!. Full ConvSDM models and baselines. Seeing what the pareto frontier is for both performance and continual learning abilities. Does SDM mess up training? and can it still learn manifolds? 
"""

settings_for_all = dict(
    epochs_to_train_for = 250, 
    dataset = DataSet.CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    num_binary_activations_for_gaba_switch=5000000,
)

name_suffix = "_CIFAR10_FullTrain" 

exp_list = [

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=10",
        use_top_k=True, 
        k_min=10,
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "ReLU_Conv_SDM_opt=SGDM",
        use_top_k=False, 
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "ReLU_Conv_SDM_opt=AdamW",
        use_top_k=False, 
        opt='AdamW',
        lr=0.01, 
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=3",
        use_top_k=True, 
        k_min=3,
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=25",
        use_top_k=True, 
        k_min=25,
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=100",
        use_top_k=True, 
        k_min=100,
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=250",
        use_top_k=True, 
        k_min=250,
    ),
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)