# simple test script to make sure that everything is workign or easy debugging: 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from py_scripts.dataset_params import *
from py_scripts.combine_params import *

model_style = ModelStyles.CLASSIC_FFN #ACTIVE_DENDRITES #FFN_TOP_K #CLASSIC_FFN #SDM
dataset = DataSet.SPLIT_MNIST

load_path = None

extras = dict(
    num_workers=0, 
    epochs_to_train_for = 25,
    epochs_per_dataset = 5,
    cl_baseline = "EWC-MEMORY",
    #normalize_n_transform_inputs = True, 
    ewc_memory_beta=0.005,
    ewc_memory_importance=20000,
    #k_min=1, 
    #num_binary_activations_for_gaba_switch = 100000,
    #cl_baseline="MAS", # 'MAS', 'EWC-MEMORY', 'SI', 'L2', '
    #dropout_prob = 0.1,
)

if load_path:
    print("LOADING IN A MODEL!!!")

model_params, model, data_module = get_params_net_dataloader(model_style, dataset, load_from_checkpoint=load_path, **extras)

wandb_logger = None #WandbLogger(project="SDMContLearning", entity="YOURENTITY", save_dir="wandb_Logger/")
model_params.logger = wandb_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("using cuda", device)
    gpu = [0]
else: 
    print("using cpu", device)
    gpu=None

# SETUP TRAINER
if model_params.load_from_checkpoint and model_params.load_existing_optimizer_state:
    fit_load_state = load_path
else: 
    fit_load_state = None

callbacks = []
# by default we will not save the model being trained unless it is in the continual learning setting. 
if model_params.investigate_cont_learning: 
    num_checkpoints_to_keep = -1 
    model_checkpoint_obj = pl.callbacks.ModelCheckpoint(
        every_n_epochs = model_params.checkpoint_every_n_epochs,
        save_top_k = num_checkpoints_to_keep,
    )
    callbacks.append(model_checkpoint_obj)
    checkpoint_callback = True 
else: 
    checkpoint_callback = False

temp_trainer = pl.Trainer(
        logger=model_params.logger,
        max_epochs=model_params.epochs_to_train_for,
        check_val_every_n_epoch=1,
        num_sanity_val_steps = False,
        enable_progress_bar = True,
        gpus=gpu, 
        callbacks = callbacks,
        checkpoint_callback=checkpoint_callback, 
        reload_dataloaders_every_n_epochs=model_params.epochs_per_dataset, 
        
        )
temp_trainer.fit(model, data_module)
wandb.finish()
