import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from types import SimpleNamespace
from .TopK_Act import Top_K

############ SDM ############
class SDMBase(nn.Module):
    def __init__(self, params: SimpleNamespace, input_size_flattened, nneurons, output_size, log_wandb_fn,curr_ep_fn, log_neuron_activations=False, module_ind=0):
        super().__init__()
        self.params = params

        self.module_ind = module_ind

        # CURRENTLY ASSUMES THAT THE TRAINING DATA STAYS CONSTANT ACROSS ALL TRAINING. 
        self.log_neuron_activations = log_neuron_activations
        self.last_checked_epoch = 0 # even if restart training it will log it as being larger. 
        if self.log_neuron_activations: 
            # used to know when an epoch is over. 
            # just going to sum over the batch
            self.activation_summer = torch.zeros( nneurons).to(self.params.device) #(self.data_size,nneurons)

        self.log_wandb = log_wandb_fn
        self.get_curr_ep = curr_ep_fn

        self.top_k = Top_K(params, nneurons, self.log_wandb, self.get_curr_ep, module_ind=module_ind)
        self.nneurons = nneurons

        # NOTE: turned these into linear layers so they are easier to work with. Can still have no bias and access their weights for normalization etc!!
        # NOTE: using default activation with the positive weights means that half of the weights will have values of 0. 

        self.fc1 = nn.Linear(input_size_flattened, nneurons, bias=params.use_bias)

        self.purkinje_layer = nn.Linear(nneurons, output_size, bias=params.use_output_layer_bias) 

        if params.granule_sparsity_percentage: 
            self.prune_granules()
        
    def enforce_l2_norm_weights(self):
        # OVERWRITING THE BASE MODEL HERE.
        # L2 norm all the neural network weights:
        if self.params.norm_addresses:
            with torch.no_grad():
                self.fc1.weight.data /= torch.norm(self.fc1.weight.data, dim=1, keepdim=True)
        if self.params.norm_values:
            with torch.no_grad():
                self.purkinje_layer.weight.data /= torch.norm(self.purkinje_layer.weight.data, dim=1, keepdim=True)

    def no_epoch_update(self):
        if self.get_curr_ep() > self.last_checked_epoch:
            self.last_checked_epoch = self.get_curr_ep()

            #print(self.purkinje_layer.bias)
            return False
        else: 
            return True

    def log_neuron_activations_fn(self, x):

        if self.no_epoch_update():
            self.activation_summer += x.detach().sum(0)
        else: 
            # OUTPUT Neuron activations: 
            self.log_wandb(
                {
                    f"SDM_Module_{self.module_ind}/fraction_dead_train_neurons": (
                    self.activation_summer < self.params.active_threshold
                    ).type(torch.float).mean(),
                }
            )
            self.activation_summer = torch.zeros( self.nneurons).to(self.params.device) #(self.data_size,nneurons)

            # output weight sparsity
            # get sparsity
            # absolute value here again to capture the potential negative weights
            weight_sparsity_percentage = (
                torch.abs(self.fc1.weight.data) < self.params.sparsity_threshold
            ).sum(1) / (self.fc1.weight.data.shape[0]*self.fc1.weight.data.shape[1])

            self.log_wandb(
                {
                    "weight_sparsity_percentage": weight_sparsity_percentage,
                }
            )

    ###### FORWARD PASS #####
    def forward(self, x, output_model_data=False):
        
        if self.params.all_positive_weights: 
            # note that none of the inputs should be negative here!
            if (x<0).sum()>0:
                pass
                #print("WARNING THERE ARE NEGATIVE VALUES IN THE INPUT WHILE ALL POSITIVE WEIGHTS ARE TURNED ON")
            x = nn.ReLU()(x) # this will help with the adversarial attacks staying positive too. 
        
        # l2 norm the input data
        if self.params.norm_addresses:
            x = x / torch.norm(x, dim=1, keepdim=True)

        x = self.fc1(x)

        x = self.top_k(x) # SDM: top-k approximated dynamic d to preserve expected neurons within radius d.

        if self.training and self.log_neuron_activations:
            self.log_neuron_activations_fn(x)

        active_values = torch.clone(x.detach())
        # num_active = (active_values > self.params.active_threshold).sum(axis=1)
        x = self.purkinje_layer(x)
      
        if output_model_data:
            model_data_dict = {
                'post_acts':active_values, 
                #'pre_acts':pre_active_values
            }
            return x, model_data_dict
        
        return x
