import numpy as np
import torch
import torch.nn as nn
import wandb
import pandas as pd
from types import SimpleNamespace
from models import BaseModel
from .TopK_Act import Top_K
import torch.nn.utils.prune as prune

############ Top-K ############
class FFN_TOP_K(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.top_k = Top_K(params, params.nneurons[0],  self.log_wandb, self.return_curr_ep)

        # cheap hack to make it work with SDM
        self.sdm_module = dict(top_k = self.top_k)
        self.sdm_module = SimpleNamespace(**self.sdm_module)

        self.fc1 = nn.Linear(params.input_size, params.nneurons[0], bias=params.use_bias)
        self.output_layer = nn.Linear(params.nneurons[0], params.output_size, bias=params.use_output_layer_bias)

    ###### FORWARD PASS #####
    def forward(self, x, log_metrics=True, output_model_data=False):
        # flatten
        if self.params.img_dim: 
            assert len(x.shape) == 4
            x = x.flatten(start_dim=1)
        # l2 norm the input data
        if self.params.norm_addresses:
            x = x / torch.norm(x, dim=1, keepdim=True)
        x = self.fc1(x)
        if self.params.use_top_k:
            x = self.top_k(x)
        else: 
            x = self.params.act_func(x)
        if output_model_data:
            active_values = torch.clone(x)

        x = self.output_layer(x)

        # for logging and other outputs. 
        if output_model_data:
            model_data_dict = {
                'post_acts':active_values, 
            }
        else: 
            return x

        return x, model_data_dict

    def enforce_l2_norm_weights(self):
        # OVERWRITING THE BASE MODEL HERE.
        # L2 norm all the neural network weights:
        #ipdb.set_trace()
        if self.params.norm_addresses:
            with torch.no_grad():
                self.fc1.weight.data /= torch.norm(self.fc1.weight.data, dim=1, keepdim=True)

