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
Oracle Model training. Swap out the dataset to get different oracle results. 
"""

settings_for_all = dict(
    epochs_to_train_for = 300, 
    dataset = DataSet.Cached_ConvMixer_CIFAR10,
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
)

name_suffix = "TrainOracle" 

exp_list = [

   dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_SGD",
        opt="SGD",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_SGD_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_Adam",
        opt="Adam",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_Adam_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "ReLU_Adam_10K",
        opt="Adam",
        load_path = "experiments/ConvMixer_ImageNet32_ImageNet32_ContLearnStarters/ReLU_Adam_10KNeurons_ContinualLearningPretrains_ConvMixer_WTransforms_ImageNet32_ImageNet32",
    ),
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)