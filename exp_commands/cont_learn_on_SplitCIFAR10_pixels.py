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
Trained on raw imagenet now testing on CIFAR10 splits. 
"""

settings_for_all = dict(
    epochs_to_train_for = 10000, 
    epochs_per_dataset = 2000,
    classification=True,
    adversarial_attacks=False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    opt="SGD",
    lr=0.05,
)

name_suffix = "_SGD_TEST_SUPERLONG_Raw_ImageNet32" 

og_exp_list = [

    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC_10K",
        nneurons=[10000],
        cl_baseline = "EWC-MEMORY",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path="experiments/ImageNet32_Raw_Pretrains/SDM_10K_LargerGABA_Pretrains_Raw_ImageNet32", 
    ),
    dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_10K", 
        nneurons=[10000], 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path="experiments/ImageNet32_Raw_Pretrains/SDM_10K_LargerGABA_Pretrains_Raw_ImageNet32", 
    ), 

    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        test_name= "Memory-EWC",
        cl_baseline = "EWC-MEMORY",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path="experiments/ImageNet32_Raw_Pretrains/ReLU_SGDM_LargerGABA_Pretrains_Raw_ImageNet32",
    ),
    dict(
        model_style= ModelStyles.SDM,
        test_name= "SDM_Memory-EWC",
        cl_baseline = "EWC-MEMORY",
        ewc_memory_beta=0.08,
        ewc_memory_importance=100,
        load_path="experiments/ImageNet32_Raw_Pretrains/SDM_LargerGABA_Pretrains_Raw_ImageNet32", 
    ),
    
   dict( 
        model_style=ModelStyles.SDM, 
        test_name="SDM_k=1", 
        k_min=1, 
        k_approach="GABA_SWITCH_ACT_BIN", 
        load_path="experiments/ImageNet32_Raw_Pretrains/SDM_k=1_LargerGABA_Pretrains_Raw_ImageNet32", 
    ), 
]

exp_list = []
for rand_seed in [3, 15, 27, 97, None]:

    for e in og_exp_list:

        temp_exp = copy.deepcopy(e)

        if rand_seed is not None: 
            temp_exp['dataset'] = DataSet[DataSet.SPLIT_CIFAR10.name +f"_RandSeed_{rand_seed}"]
        else: 
            temp_exp['dataset'] = DataSet.SPLIT_CIFAR10

        temp_exp['test_name'] = temp_exp['test_name']+f"_RS_{rand_seed}"

        exp_list.append(temp_exp)


if __name__ == '__main__':
    print(len(exp_list))
    sys.exit(0)