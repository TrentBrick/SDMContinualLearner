import copy
from enum import Enum, auto
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from models import *
from torch import nn
import torchvision.models as models

from py_scripts import LightningDataModule
from .model_params import *
from .dataset_params import *
import ipdb

# consider using https://pypi.org/project/easydict/ instead of simplenamespace? probably not worth it at this point though. 

def reload_params_from_saved_model(full_cpkt, params, custom_kwargs, dataset_params ):

    del full_cpkt["hyper_parameters"]['model_foundation']
    
    params = copy.deepcopy(params)
    dataset_params = copy.deepcopy(dataset_params)
    # have experimental settings overwrite those from the model loaded in. 
    custom_kwargs = copy.deepcopy(custom_kwargs)

    # this will ensure that the custom params dominate
    full_cpkt["hyper_parameters"].update(dataset_params)
    full_cpkt["hyper_parameters"].update(custom_kwargs)
    
    # update parameters with model loaded in ones. 
    params.update(full_cpkt["hyper_parameters"])

    return params

def get_params_net_dataloader(
    model_style: ModelStyles,
    dataset: DataSet,
    load_from_checkpoint=None,
    dataset_path="data/",
    verbose=True,
    cont_learn_reset_output_head=True,
    **custom_kwargs,
):
    """
    Returns model parameters for given model_style and an optional list of regularizers. 
    """
    #if load_from_checkpoint is None:
    params = default_settings
    params["dataset"] = dataset
    params["dataset_str"] = dataset.name
    params['model_style'] = model_style

    # should probably have as own dictionaries that are updated.
    ########### DATASET CONFIGS ###########
    dataset_params = assign_dataset_params(dataset)
    params.update(dataset_params)

    # used for the custom faster torch datasets. 
    dataset_path+= dataset_params['dataset_path_suffix']

    ######### PARAMETER CUSTOMIZATION ##########
    model_params = assign_model_params(model_style)
    params.update(model_params)
    
    params["act_string"] = str(params["act_func"])
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up here so it applies automatically and is not over written by any loaded in model (custom kwards overwrites the loaded in model parameters.)

    #NOTE: this is kind of hacky but helps ensure no errors during the run because I forget to set the wrong master parameters.
    if "CachedOutputs" in params['dataset_path_suffix']:
        params['log_receptive_fields'] = False
        params['log_model_predictions'] = False

    custom_kwargs['continual_learning']=True if "SPLIT" in dataset.name else False
    custom_kwargs["dataset_str"] = dataset.name

    # Running this first and then overwriting with it again later!!!
    #-----------------
     # implement custom_kwargs and do so before set the model and dataset.
    if verbose: 
        print("Custom args are:", custom_kwargs)
    for key, value in custom_kwargs.items():
        params[key] = value

    ###########

    if "k_transition_epochs" in params.keys() and params["k_transition_epochs"] is None:
        params["k_transition_epochs"] = int(params["epochs_to_train_for"] / 2)

    ##########
    # loading in model
    if load_from_checkpoint:  
        import os
        print(load_from_checkpoint, os.getcwd())
        full_cpkt = torch.load(load_from_checkpoint, map_location=params["device"])

        # pure reload 
        params = reload_params_from_saved_model(full_cpkt, params, custom_kwargs, dataset_params )

    # ensure the device is not overwritten no matter what. 
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['load_from_checkpoint'] = load_from_checkpoint
        
    # general logic based processing: 
    if params['continual_learning']:
        #TODO: this is a bug () but it is one I need to account for for now!!! 
        params['check_val_every_n_epoch'] = 1
        
        if not params['investigate_cont_learning']:
            params['save_model_checkpoints'] = False 
    
    params["output_size"] = params["nclasses"]

    params = SimpleNamespace(**params)
    data_module = LightningDataModule(
        params,
        data_path=dataset_path
    )

    if load_from_checkpoint is not None:
        # load in the original model but iwth its output and input parameters.
        # TODO: I really need to clean this up!
        load_params_change_output_size = copy.deepcopy(params) 
        load_params_change_output_size.output_size =full_cpkt['hyper_parameters']['output_size']
        model = params.model_foundation(load_params_change_output_size)
    else: 
        model = params.model_foundation(params)
    
    # Make modifications to the model
    if load_from_checkpoint is not None:
        if verbose: 
            print("!!! Loading in model with checkpoint saved parameters!!!")

        model.load_state_dict( full_cpkt['state_dict'])

        # needs to increase the number of epochs_to_train_for further. 
        #params.epochs_to_train_for += custom_kwargs['epochs_to_train_for']

        if "epoch" in full_cpkt:
            model.curr_ep = full_cpkt["epoch"]
            params.starting_epoch=full_cpkt["epoch"]

        if params.continual_learning:
            # pure reload but modify a head
            params.load_existing_optimizer_state = False

            # reset the output head of the ImageNet32 model. 
            
            print("CHANGING OUTPUT LAYER FOR CONTINUAL LEARNING ON NEW DATASET!")
            if cont_learn_reset_output_head:
                # this is only false when I am trying to load in and analyze a trained model. 

                if params.model_foundation == SDM:
                    model.sdm_module.purkinje_layer = nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_output_layer_bias)
                #elif params.model_foundation == ConvMixer:
                #    nn.Linear(params.hdim, params.output_size)
                elif params.model_foundation == CLASSIC_FFN or params.model_foundation == FFN_TOP_K:
                    model.net[-1] = nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_output_layer_bias)
                elif params.model_foundation == ConvSDM:
                    model.sdm_module.purkinje_layer = nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_output_layer_bias)
                else: 
                    model.net[-1] = nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_output_layer_bias)

        if params.load_existing_optimizer_state and "epoch" in full_cpkt:
            params.epochs_to_train_for += params.starting_epoch

        del full_cpkt

    if verbose: 
        print(
            "Number of unique parameters trained in the model",
            len(list(model.parameters())),
        )

    non_zero_weights = 0
    for p in list(model.parameters()):
        if len(p.shape)>1: 
            non_zero_weights+= (torch.abs(p)>0.0000001).sum()
    if verbose: 
        print("Number of non zero weights is:", non_zero_weights)

    #params.train_data_size = len(data_module.train_data)
    #params.test_data_size = len(data_module.test_data)

    if params.log_receptive_fields_every_n_epochs>1:
        assert params.log_receptive_fields_every_n_epochs%params.check_val_every_n_epoch==0, "need for overall validation logging interval to be a fraction of the receptive field logger. Else will never log!"

    if verbose: 
        print("Final params being used", params)

    # just really making sure these are the same thing. 
    model.params = params

    return params, model, data_module