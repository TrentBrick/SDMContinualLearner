import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from types import SimpleNamespace
from models import BaseModel, Residual
from .SDM_Base import SDMBase
import ipdb

############ SDM ############
class SDM(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)
        
        self.sdm_module = SDMBase(params, params.input_size, params.nneurons[0], params.output_size, self.log_wandb, self.return_curr_ep, log_neuron_activations= params.log_for_dead_neurons)

    def enforce_l2_norm_weights(self):
        # need to pass on this function call from Base_Model.py
        self.sdm_module.enforce_l2_norm_weights()

    ###### FORWARD PASS #####
    def forward(self, x, output_model_data=False):
        
        if self.params.img_dim: 
            assert len(x.shape) == 4
            # if no image dim it means that it has been cached via another model. 
        
        x = x.flatten(start_dim=1)

        if self.params.count_dead_train_neurons or self.params.log_gradients:
            # override any other inputs from either train or validation. 
            output_model_data = True
            
        x = self.sdm_module(x, output_model_data=output_model_data)
        
        if output_model_data:
            x, model_data_dict = x

        # for logging and other outputs. 
        if output_model_data:
            return x, model_data_dict
            
        return x