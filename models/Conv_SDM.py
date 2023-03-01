# ConvMixer from Patches are all you need with timm training augmentations: 
# https://github.com/locuslab/convmixer-cifar10/blob/main/train.py


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from types import SimpleNamespace
from models import BaseModel, Residual
from .SDM_Base import SDMBase
import numpy as np
import time

"""Just replacing the last layer with the SDM module. """

class ConvSDM(BaseModel):
    def __init__(self, params: SimpleNamespace):
        super().__init__(params)

        self.features = nn.Sequential(
            nn.Conv2d(3, params.hdim, kernel_size=params.patch_size, stride=params.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(params.hdim),
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(params.hdim, params.hdim, params.kernel_size, groups=params.hdim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(params.hdim)
                    )),
                    nn.Conv2d(params.hdim, params.hdim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(params.hdim)
            ) for i in range(params.nblocks)],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if params.use_top_k:
            self.sdm_module = SDMBase(params, params.hdim, params.nneurons[0], params.output_size, self.log_wandb, self.return_curr_ep,  log_neuron_activations= params.log_for_dead_neurons)
        else:
            self.sdm_module = SDM_Module_ReLU_Replacement(params.hdim, params.nneurons[0], params.output_size)

    def enforce_l2_norm_weights(self):
        self.sdm_module.enforce_l2_norm_weights()
        
    def forward(self, x, output_model_data=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.sdm_module(x, output_model_data=output_model_data)
        if output_model_data:
            x, model_data_dict = x
            return x, model_data_dict

        return x

class SDM_Module_ReLU_Replacement(nn.Module):
    def __init__(self, input_size_flattened, nneurons, output_size):
        super().__init__()
        self.inp = nn.Linear(input_size_flattened, nneurons)
        self.relu = nn.ReLU()
        self.out = nn.Linear(nneurons, output_size)

    def enforce_l2_norm_weights(self):
        pass

    def forward(self, x, output_model_data=False):
        x = self.relu(self.inp(x))
        active_values = torch.clone(x)
        x = self.out(x)
        
        # for logging and other outputs. 
        if output_model_data:
            model_data_dict = {
                'post_acts':active_values, 
                #'pre_acts':pre_active_values
            }
            return x, model_data_dict

        return x