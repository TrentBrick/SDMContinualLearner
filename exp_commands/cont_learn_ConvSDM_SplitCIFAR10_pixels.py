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
Testing some 10K conv SDMs on continual learning as a proof of principle. 
"""

settings_for_all = dict(
    epochs_to_train_for = 1000, 
    dataset = DataSet.SPLIT_CIFAR10, 
    classification=True,
    adversarial_attacks=False, 
    epochs_per_dataset = 200,
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    use_convmixer_transforms = False, 
    opt="SGDM",
    lr=0.03,

    cl_baseline = "EWC-MEMORY",
    ewc_memory_beta=0.08,
    ewc_memory_importance=300,
)

name_suffix = "_wEWC_TEST_Raw_Split_CIFAR10-Subtract_imp=300" 

exp_list = [

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=100-Mask",
        use_top_k=True, 
        k_min=100,
        k_approach = "LINEAR_DECAY_SUBTRACT",
        load_path ="experiments/Conv_SDM_ImageNet32/Conv_SDM_k=100_10K_ImageNet_FullTrain_Longer-Mask",
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=1000-Mask",
        use_top_k=True, 
        k_min=1000,
        k_approach = "LINEAR_DECAY_SUBTRACT",
        load_path ="experiments/Conv_SDM_ImageNet32/Conv_SDM_k=1000_10K_ImageNet_FullTrain_Longer-Mask",
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=2500-Mask",
        use_top_k=True, 
        k_min=2500,
        k_approach = "LINEAR_DECAY_SUBTRACT",
        load_path ="experiments/Conv_SDM_ImageNet32/Conv_SDM_k=2500_10K_ImageNet_FullTrain_Longer-Mask",
    ),

    dict( 
        model_style=ModelStyles.CONV_SDM, 
        test_name="Conv_SDM_k=100", 
        use_top_k=True, 
        k_min=100, 
        load_path="experiments/Conv_SDM_CIFAR10/Conv_SDM_k=100_10K_ImageNet_FullTrain_Longer",
    ), 

    dict( 
        model_style=ModelStyles.CONV_SDM, 
        test_name="Conv_SDM_k=250", 
        use_top_k=True, 
        k_min=250, 
        load_path="experiments/Conv_SDM_CIFAR10/Conv_SDM_k=250_10K_ImageNet_FullTrain_Longer",
    ), 
   
   dict( 
        model_style=ModelStyles.CONV_SDM, 
        test_name="Conv_SDM_k=1000", 
        use_top_k=True, 
        k_min=1000, 
        load_path="experiments/Conv_SDM_CIFAR10/Conv_SDM_k=1000_10K_ImageNet_FullTrain_Longer", 
    ), 

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "Conv_SDM_k=2500",
        use_top_k=True, 
        k_min=2500,
        load_path="experiments/Conv_SDM_CIFAR10/Conv_SDM_k=2500_10K_ImageNet_FullTrain_Longer",
    ),

    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "ReLU_Conv_SDM_opt=SGDM",
        use_top_k=False, 
        load_path="experiments/Conv_SDM_CIFAR10/ReLU_Conv_SDM_opt=SGDM_10K_ImageNet_FullTrain_Longer",
    ),
    dict(
        model_style= ModelStyles.CONV_SDM,
        test_name= "TopK_Conv_SDM_opt=SGDM",
        use_top_k=True, 
        k_approach = "LINEAR_DECAY_MASK",
        use_bias=True,
        all_positive_weights=False,
        norm_addresses=False,
        load_path="experiments/Conv_SDM_CIFAR10/TopK_Conv_SDM_opt=SGDM_10K_ImageNet_FullTrain_Longer",
    ),
    
]

if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)