import numpy as np
import torch
import torch.nn as nn
import wandb
import pandas as pd
from types import SimpleNamespace
from models import BaseModel

############ Classic FFN ############
class CLASSIC_FFN(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.fc1 = nn.Linear(params.input_size, params.nneurons[0])

        if params.dropout_prob>0:
            self.fc1_dropout = nn.Dropout(p=params.dropout_prob)

        '''self.net = nn.Sequential( 
            *[nn.Sequential(nn.Linear(params.nneurons[i], params.nneurons[i+1]), params.act_func) for i in range(len(params.nneurons)-1) ], nn.Linear(params.nneurons[-1], params.output_size)
        )'''
        self.output_layer = nn.Linear(params.nneurons[-1], params.output_size, bias=params.use_output_layer_bias)

    ###### FORWARD PASS #####
    def forward(self, x, log_metrics=False, output_model_data=False):
        # flatten
        if self.params.img_dim: 
            assert len(x.shape) == 4
            x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.params.act_func(x)
        if self.params.dropout_prob>0:
            x = self.fc1_dropout(x)
        if output_model_data:
            active_values = torch.clone(x)
        #x = self.net(x)
        x = self.output_layer(x)
        
        if output_model_data:
            model_data_dict = {
                'post_acts':active_values, 
            }
            return x, model_data_dict
        
        return x