# Basic vanilla FFN that can also implement TopK. 

import numpy as np
import torch
import torch.nn as nn
import wandb
import pandas as pd
from types import SimpleNamespace
from models import BaseModel

#from nupic.research.frameworks.dendrites import DendriticMLP 

class InterfActiveDendrites(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.params = params
        act_dend_params = dict(
            
            input_size=params.input_size,
            output_size=params.output_size,  # Single output head shared by all tasks
            hidden_sizes=params.hidden_sizes,
            dim_context=params.input_size,
            num_segments = params.nclasses,
            kw=params.kw,
            kw_percent_on=params.kw_percent_on,
            dendrite_weight_sparsity=params.dendrite_weight_sparsity,
            weight_sparsity=params.weight_sparsity,
            context_percent_on=params.context_percent_on,
        )

        self.actual_model = DendriticMLP(**act_dend_params)
        if "ConvMixerWTransforms" in params.dataset_str and "CIFAR10" in params.dataset_str:
            self.context_vectors = torch.load("data/context_vectors/ConvMixerWTransforms_ImgNet32_CIFAR10.pt").to(params.device)
        elif "MNIST" in params.dataset_str:
            self.context_vectors = torch.load("data/context_vectors/MNIST.pt").to(params.device)
        else:
            raise NotImplementedError("Dont know this.")

    ###### FORWARD PASS #####
    def forward(self, x, task_info=None, log_metrics=False, output_model_data=False):
        # flatten
        if self.params.img_dim: 
            assert len(x.shape) == 4
            x = x.flatten(start_dim=1)

        if task_info is not None: 
            # context encoding
            task_info = self.context_vectors[task_info]
            # one hot encoding
            #task_info = torch.zeros((len(task_info), self.params.nclasses)).scatter(1, task_info.unsqueeze(1), 1)
            
        x= self.actual_model(x, context=task_info)

        if output_model_data:
            model_data_dict = {
                'post_acts':None, 
                #'pre_acts':pre_active_values
            }
            return x, model_data_dict
        
        return x