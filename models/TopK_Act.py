import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Top_K(nn.Module):
    def __init__(self, params, nneurons, log_wandb_fn,curr_ep_fn, module_ind=0):
        super().__init__()
        self.params = params
        self.ReLU = nn.ReLU()
        self.log_wandb = log_wandb_fn
        self.get_curr_ep = curr_ep_fn
        self.module_ind = module_ind
        if params.k_max is None:
            self.k_max = nneurons
        else: 
            assert nneurons<=params.k_max, "K max is too large for this layer"
        
        # track activations for GABA
        if self.params.k_approach == "GABA_SWITCH_ACT_BIN":
            self.neuron_activation_counters = torch.zeros((1,nneurons) , requires_grad=False).to(params.device)
            self.linear_coef_threshold = self.params.num_binary_activations_for_gaba_switch

    def get_curr_k(self):
        # linearly (across epochs_to_train_for) reduce k value between k_max and k_min
        if "LINEAR_DECAY" in self.params.k_approach:
            k_max = self.k_max
            linear_coef = (
                -(k_max - self.params.k_min) / self.params.k_transition_epochs
            )
            k = np.minimum(
                k_max,
                np.maximum(
                    k_max + (linear_coef * self.get_curr_ep()), self.params.k_min,
                ),
            )
            k = int(k)
        else:
            k = self.params.k_min

        if "GABA_SWITCH" in self.params.k_approach:
            k=k+1 # to subtract by this value
            # +1 because we are finding the value for this lowest one to then subtract from. 
        return k

    def forward(self, x, k_dim=1, layer_ind=None):
        x = self.ReLU(x)

        unweighted_input_into_inhib_neuron = torch.sum(x, dim=k_dim).detach()
        curr_k = self.get_curr_k()
        vals, inds = torch.topk(x, np.minimum(curr_k, self.k_max), dim=k_dim, sorted=False)
        inhib_amount, _ = torch.min(vals.detach(), dim=k_dim, keepdim=True)

        if "GABA_SWITCH" in self.params.k_approach:
            # get threshold coefficient
            linear_coef = 2 / self.linear_coef_threshold
            gaba_response = torch.minimum(
                torch.ones_like(self.neuron_activation_counters),
                torch.maximum(
                    -1 + (linear_coef * self.neuron_activation_counters),
                    torch.ones_like(self.neuron_activation_counters) * -1,
                ),
            ).type_as(x)

            if self.params.log_metrics is True:
                if layer_ind is None or layer_ind==self.params.gaba_switch_logging_layer:
                    self.log_wandb( {
                            f"TopK_Act_{self.module_ind}/neuron_binary_activation_counters": self.neuron_activation_counters,
                            f"TopK_Act_{self.module_ind}/gaba_response": gaba_response
                        })
        else:
            gaba_response = 1

        # apply inhibition
        if "MASK" in self.params.k_approach:
            top_k_mask = torch.zeros_like(x)
            top_k_mask = top_k_mask.scatter(k_dim, inds, 1)
            x = x * top_k_mask 
        else: 
            # GABA switch or Subtracts
            x = self.ReLU(x - (gaba_response * inhib_amount))
            self.neuron_activation_counters += torch.sum(x>0, dim=0, keepdim=True)

        if self.params.log_metrics is True:
            self.log_wandb(
                {
                    f"TopK_Act_{self.module_ind}/unweighted_input_into_inhib_neuron_per_batch": unweighted_input_into_inhib_neuron,
                    f"TopK_Act_{self.module_ind}/inhib_amount_per_batch": inhib_amount.flatten(),
                    f"TopK_Act_{self.module_ind}/k": curr_k,
                }
            )

        return x 