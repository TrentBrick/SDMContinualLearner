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
ImageNet32 pixels. 
10K Neurons and trying a number of k values for the joint training of the ConvSDM models. Seeing what the pareto frontier is for both performance and continual learning abilities. Does SDM mess up training? And can it still learn manifolds? 
"""

settings_for_all = dict(
    epochs_to_train_for = 30, 
    dataset = DataSet.ImageNet32, 
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    nneurons=[10000],
    num_binary_activations_for_gaba_switch=10000000,
    k_approach = "LINEAR_DECAY_MASK",
    k_transition_epochs = 10,
    checkpoint_every_n_epochs = 1,
)

name_suffix = "_10K_ImageNet_FullTrain_Longer" 

exp_list = [

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=100",
        use_top_k=True, 
        k_min=100,
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=250_PosWeights",
        use_top_k=True, 
        k_min=250,
        all_positive_weights=True,
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=1000",
        use_top_k=True, 
        k_min=1000,
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=2500",
        use_top_k=True, 
        k_min=2500,
    ),
    
    
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)