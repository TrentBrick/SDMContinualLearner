import copy
from enum import Enum, auto
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from models import *
from torch import nn

from py_scripts import LightningDataModule

# consider using https://pypi.org/project/easydict/ instead of simplenamespace? probably not worth it at this point though. 

class ModelStyles(Enum):
    """
    Enum for the different model styles.
    """
    # replications:
    CLASSIC_FFN = auto()
    CONVMIXER = auto()
    FFN_TOP_K = auto()
    ACTIVE_DENDRITES =auto()
    # new models
    SDM = auto()
    CONV_SDM = auto()
    
def assign_model_params(model_style):
    model_params = {}
    if model_style is ModelStyles.CLASSIC_FFN:
        model_params.update( classic_ffn_settings )
        model_params["model_foundation"] = CLASSIC_FFN
    elif model_style is ModelStyles.SDM:
        model_params.update(default_top_k_sdm_fnn_settings)
        model_params.update(sdm_settings)
        model_params["model_foundation"] = SDM
    elif model_style is ModelStyles.FFN_TOP_K:
        model_params.update(default_top_k_sdm_fnn_settings)
        model_params.update(ffn_top_k)
        model_params["model_foundation"] = FFN_TOP_K
    elif model_style is ModelStyles.CONVMIXER:
        model_params.update(convmixer_settings)
        model_params["model_foundation"] = ConvMixer
    elif model_style is ModelStyles.CONV_SDM:
        model_params.update(default_top_k_sdm_fnn_settings)
        model_params.update(sdm_settings)
        model_params.update(convmixer_settings)
        model_params.update(conv_sdm_settings)
        model_params["model_foundation"] = ConvSDM
    elif model_style is ModelStyles.ACTIVE_DENDRITES:
        model_params.update(active_dendrites_settings)  
        model_params["model_foundation"] = InterfActiveDendrites
    else:
        print("Errored model style is:",model_style)
        raise NotImplementedError(model_style)

    return model_params

default_settings = dict(
    
    # general default settings that seem reasonable. 
    investigate_cont_learning =False,  
    act_func=nn.ReLU(),
    batch_size=128,
    epochs_to_train_for=200,
    num_workers=2, # for the data loader
    starting_epoch = 0, 
    load_existing_optimizer_state = True, # else will only load in the model weights. 
    use_top_k=False, 
    use_bias = True, 
    use_output_layer_bias = True, 
    using_vae=False, 
    norm_addresses=False,
    norm_values=False, 
    all_positive_weights = False, 
    
    cl_baseline = None, 
    cl_baseline_batches_per_dataset = 5,
    cl_baseline_batch_size = 512,

    mas_importance=0.5, 
    si_importance=1500, 
    si_beta = 0.005,
    ewc_memory_importance=200, 
    ewc_memory_beta=0.005,
    l2_importance = 10,

    # continual learning
    epochs_per_dataset = 100,

    # default optimizer settings
    opt="SGDM",  # Stochastic Grad Descent with Momentum. # Adam
    lr=0.03,
    sgdm_momentum=0.9,

    gradient_clip=1.0,
    adamw_l2_loss_weight=0.001,

    # other training features/options
    normalize_n_transform_inputs =False, 
    use_convmixer_transforms=False,

    # for logging/analysis
    check_val_every_n_epoch = 1, # how often to run the validation loop
    save_model_checkpoints = True, 
    checkpoint_every_n_epochs = 10,
    log_metrics = True, 
    log_model_predictions = True, 
    validation_neuron_logger = True, # log things about the neurons
    log_receptive_fields = False, # log neuron receptive fields as specified. Only works for SDM_FFN for now. 
    log_receptive_fields_every_n_epochs = 10, # trying to reduce memory consumption. 


    log_gradients=False,
    start_epoch_log_grads =95,


    log_for_dead_neurons = True,

    active_threshold=0.0001,
    sparsity_threshold=0.0001,
    num_receptive_field_imgs=10,
    
    dropout_prob=0.0,

    # ConvMixer processing parameters
    scale=0.75,
    reprob=0.25, 
    ra_m=8,
    ra_n=1,
    jitter=0.1,

    # others:
    can_enforce_positive_weights_on_bias_terms = True, 
    epoch_relative_to_training_restart = False,
)

classic_ffn_settings = dict(
    all_positive_weights=False,
    nneurons=[1000],
)

# the following all share the same SDM FFN Model class. 
default_top_k_sdm_fnn_settings = dict(
    # shared by all of the settings that follow
    nneurons=[1000],
    use_top_k=True,
    k_max=None,  # what k starts out as. should probably be 1000? Frey uses 100.
    k_min=10,
    k_transition_epochs = 50,
    num_binary_activations_for_gaba_switch=500000, #10000000,  # number of activations to switch gaba from -1*activation to 1*activation
    num_activations_for_gaba_switch = 900000, #3000000 #300000
    num_grads_for_gaba_switch = 5000,
    granule_sparsity_percentage = None, # if the input weights should be made sparse at all.
    all_positive_weights=False,
    norm_addresses=False,  
)

ffn_top_k = dict(
    k_approach = "LINEAR_DECAY_MASK", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
)

# SDM params. Single layer model
sdm_settings = dict(
    k_approach = "GABA_SWITCH_ACT_BIN", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH"
    # Network deviations from MLP
    use_bias=False,
    use_output_layer_bias = False, 
    all_positive_weights=True,
    norm_addresses=True,  # L2 norm of neuron addresses and inputs.
    learn_addresses=True,
    norm_values=False, # also means no betas. for the output weights.
    relu_exponent = None, # can use higher order exponents a la Krotov. None means no higher power. Needs to be a float. 
    count_dead_train_neurons=False,
    unique_weight_init=False, # ensuring that none of the weights start off negative. 
    purkinje_layer = False, 
    npurkinje = 100,
    batch_size=128,
    lr=0.03,

    # CNN layer
    apply_cnn_layer = False,
    cnn_relu=True, 
    # need to see if having normalization on then helps or not. 
    cnn_nfilters = 128, 
    cnn_kern_size = 4,
    cnn_stride = None, # None means same as kern size.
    cnn_padding = 0, 
)

convmixer_settings = dict(
    # not implementing learning rate schedule for now. 
    # or the gradient scaler. 
    batch_size = 512,
    #lr_schedule_triangular =True, 
    kernel_size=5, 
    patch_size=2,
    gradient_clip=1.0,
    hdim=256,
    nblocks=8,
    adamw_l2_loss_weight=0.005,
    lr=0.01, 
    opt="AdamW",
    use_convmixer_transforms=False, 
    validation_neuron_logger=False,
    log_receptive_fields=False, 

    scale=1.0,
    reprob=0.0, 
    ra_m=12,
    ra_n=2,
    jitter=0.0,
)

conv_sdm_settings =dict(
    # overwriting opt and setting shorter gaba swwitch
    k_approach = "GABA_SWITCH_ACT_BIN", 
    nblocks=2,
    num_binary_activations_for_gaba_switch=500000,
    opt='SGDM',
    lr=0.03,
    use_convmixer_transforms = False, 
    validation_neuron_logger = True,
    log_metrics = True,
    log_receptive_fields = False,
    batch_size = 128,
    
    all_positive_weights=False,
    use_top_k=True, 
)

active_dendrites_settings = dict(

    hidden_sizes=[2048, 2048],
    validation_neuron_logger=False, 
    kw=True,
    kw_percent_on=0.05,
    dendrite_weight_sparsity=0.0,
    weight_sparsity=0.5,
    context_percent_on=0.1,
    batch_size=256,
    #val_batch_size=512,
    #tasks_to_validate=[1, 4, 9, 24, 49, 99],
    opt="Adam", # should try SGD too
    lr =5e-4, 
)

