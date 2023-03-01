import wandb 
from py_scripts.dataset_params import DataSet
from py_scripts.model_params import ModelStyles
import sys 
config_dict={
    "name": "EWC_MEMORY_ContLearnTest_Pretrained_HyperparamSearch",
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "method": "grid", # 'random', 'grid', or 'bayes'
}
# Careful with value vs values!
sweep_params = {
    'epochs_to_train_for':{
        "value": 2500 # if less than 500 results may not be as meaningful
    },
    'epochs_per_dataset':{
        "value": 500 # if less than 500 results may not be as meaningful
    },
    "model_style": {
        "value": ModelStyles.CLASSIC_FFN.name
    },
    "dataset": {
        "value": DataSet.SPLIT_Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10.name
    },
    "load_path": {
        "value": "experiments/Baseline_Ablations_ConvMixer_ImgNet32_ImgNet32_ContLearnStarters/EWC_SGD_Baselines_Ablation_PRETRAIN_ConvMixer_WTransforms_ImageNet32_ImageNet32"
    },
    "adversarial_attacks": {
        "value": False
    },
    "log_receptive_fields": {
        "value": False
    },
    "log_metrics": {
        "value": True
    },
    "cl_baseline": {
        "value": "EWC-MEMORY"
    },
    "ewc_memory_beta": {
        "values": [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1.0]
    },
    "ewc_memory_importance": {
        "values": [10, 100, 500, 1000, 2000, 3000, 5000, 10000]
    },
    
}

if __name__ == '__main__':
    config_dict["parameters"] = sweep_params
    sweep_id = wandb.sweep(config_dict, project="Foundational-SDM", entity="YOURENTITY")
    print(sweep_id)
    sys.exit(0)