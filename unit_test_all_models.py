
import numpy as np 
import random 
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from py_scripts.dataset_params import *
from py_scripts.combine_params import *

# UNIT testing a bunch of experiments. 

seed = 27

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

########### EXPERIMENTS TO RUN ##############

settings_for_all = dict(
    #epochs_to_train_for = 3,
    classification=True,
    adversarial_attacks=False, 
    log_metrics = False, 
    validation_neuron_logger = False, # log things about the neurons
    log_model_predictions = False,
    log_receptive_fields = False,
    #count_dead_train_neurons=True, 
    #investigate_cont_learning = True,
    #investigate_cont_learning_log_every_n_epochs_to_train_for = 10, 
)

# TODO: Turn off wandb logger. 

name_suffix = "_UnitTests" # can set to None. 
num_batches = 30

exp_list = [
    # MNIST TESTS OF NEW ARCHITECTURES FIRST
    dict(
        model_style= ModelStyles.SDM,
        dataset = DataSet.MNIST,
        test_name= "MNIST_SDM",
        k_approach = "GABA_SWITCH_ACT_BIN", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
        test_acc = 0.11542,
    ),
    dict(
        model_style= ModelStyles.SDM,
        dataset = DataSet.MNIST,
        test_name= "MNIST_SDM",
        k_approach = "LINEAR_DECAY_MASK", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
        test_acc = 0.11643
    ),
    dict(
        model_style= ModelStyles.CLASSIC_FFN,
        dataset = DataSet.MNIST,
        test_name= "MNIST_CLASSIC_FFN",
        test_acc = 0.11164,
    ),
    dict(
        model_style= ModelStyles.FFN_TOP_K,
        dataset = DataSet.MNIST,
        test_name= "MNIST_FFN_TOP_K",
        k_approach = "GABA_SWITCH_ACT_BIN", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
        test_acc = 0.09879, 
    ),
]

def train_loop(params, model, data_module, device):

    data_module.setup(None, train_shuffle=False, test_shuffle=False)
    train_loader = data_module.train_dataloader()
    test_loader = data_module.val_dataloader()
    for train_o_test, loader in zip(['Train', 'Test'], [train_loader, test_loader]):
        with torch.no_grad(): 
            accuracies = []
            for dind, data in enumerate(loader):
                if dind>num_batches:
                    break 
                #if dind >0:
                #    print( 'batch ind:', dind,  'curr_accs', round(np.mean(accuracies),5))
                x, y = data
                x, y = x.to(device), y.to(device)
                out = model.forward(x, output_model_data=params.validation_neuron_logger)
                if params.validation_neuron_logger:
                    logits, model_data_dict = out
                    acts = model_data_dict['post_acts']
                else:
                    logits = out

                b_accuracies = (logits.argmax(dim=1)==y)
                #loss = model.compute_loss(logits, y, x)
                accuracies += list(b_accuracies.cpu().numpy())
            print(train_o_test," accuracy", np.mean(accuracies))

    return round(np.mean(accuracies),5)

load_path = None

for exp in exp_list:

    model_style = exp["model_style"]
    exp.pop("model_style")
    dataset = exp["dataset"]
    exp.pop("dataset")

    exp.update(settings_for_all)
    model_params, model, data_module = get_params_net_dataloader(model_style, dataset, load_from_checkpoint=load_path, **exp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_params.logger = None

    test_acc = train_loop(model_params, model, data_module, device)

    if exp['test_acc'] == test_acc: 
        print("="*10)
        print("="*10)
        print(exp['test_name'], "replicates!")
        print("With parameters:", exp)
        print("="*10)
        print("="*10)
    else: 
        raise Exception(exp['test_name'])


####################################

"""

dict(
        model_style= ModelStyles.DEEP_SDM,
        test_name= "MNIST_DEEP_SDM",
        k_approach = "LINEAR_DECAY_MASK", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    ),
    dict(
        model_style= ModelStyles.DEEP_SDM,
        test_name= "MNIST_DEEP_SDM",
        k_approach = "GABA_SWITCH", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    ),
    dict(
        model_style= ModelStyles.MICROZONES,
        test_name= "MNIST_MICROZONES",
        k_approach = "GABA_SWITCH", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    ),
    dict(
        model_style= ModelStyles.CONVNEXT,
        test_name= "MNIST_CONVNEXT",
    ),
    dict(
        model_style= ModelStyles.ALEX_NET,
        test_name= "MNIST_ALEX_NET",
    ),
    dict(
        model_style= ModelStyles.CONVMIXER,
        test_name= "MNIST_CONVMIXER",
    ),
    dict(
        model_style= ModelStyles.ALEX_SDM,
        test_name= "MNIST_ALEX_SDM",
        k_approach = "GABA_SWITCH", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    ),
    dict(
        model_style= ModelStyles.PRODUCT_KEY,
        test_name= "MNIST_PRODUCT_KEY",
        k_approach = "GABA_SWITCH", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    ),

    # TRYING TO LOAD IN MODELS

    dict(
        model_style= ModelStyles.ALEX_SDM,
        test_name= "Pretrained_Frozen_ALEX_SDM_ReLU",
        use_top_k=False, 
        load_path = "experiments/AlexNet/AlexNet_NoDataAugs",
        alex_net_freeze_layer_swap = True,
    ),




"""