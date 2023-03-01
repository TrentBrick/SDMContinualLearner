import numpy as np
import torch
from torch import nn
import copy
import math
from .config import DEVICE

#Place all dataset to GPU
def taskset2GPUdataset(task_set):
    return task_set
    """
    try:
        task_set.get_sample(0)
    except:
        if task_set.nb_classes == 33 or task_set.nb_classes == 30:
            return torch.utils.data.TensorDataset(torch.tensor(task_set.get_raw_samples()[0]).to(DEVICE),
                                                  torch.tensor(task_set.get_raw_samples()[1]).to(DEVICE))
    x = []
    y = []
    for (data, target, _) in task_set:
        x.append(data)
        y.append(target)
        
    return torch.utils.data.TensorDataset(torch.stack(x).to(DEVICE), torch.tensor(y).to(DEVICE))
    """


def cosine_anneling(t, k):
    return 0.5 * (1 + math.cos(t * math.pi / k))
def linear(t, k):
    return 1 - k*t
def exp_decay(t, k):
    return (t + 1)**(-k)

# Given stable units, find the plastic units
def compute_plastic(model, stable_indices):
    weight = list_of_weights(model)
    units = [w[1].shape[0] for w in weight]
    plastic_indices = []
    for i,  b_layer in enumerate(stable_indices[1:]):
        all_units = set(range(units[i]))
        plastic_units = all_units.difference(set(b_layer))
        plastic_indices.append(list(plastic_units))
    return [[]] + plastic_indices

#Get sparse connecivity masks
def  get_masks(model):
    l = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d) :
            weight_mask, _ = module.get_mask()
            l.append(weight_mask)
    return l

#Find stable untis and drop connections
def compute_stable_and_drop_connections(pruned_model, in_channels, stable_units_set, train_loader, current_classes, classes_so_far, stable_activation_perc, model_type):
    stable_units_set = copy.deepcopy(stable_units_set)
    pruned_model = copy.deepcopy(pruned_model)
    activation_before_drop, _ = compute_average_activation(pruned_model, train_loader,
                                                            DEVICE, current_classes, False)
    
    stable_indices_new = compute_stable_neurons(activation_before_drop,  in_channels, classes_so_far, activation_perc = stable_activation_perc, model_type = model_type)

    if len(stable_units_set) == 0:
        stable_units_set = [set(stable_index) for stable_index in stable_indices_new]
        stable_indices =  [list(stable_set) for stable_set in stable_units_set]
    else:
        stable_units_set = [prev_set.union(set(stable_index)) for stable_index, prev_set in zip(stable_indices_new, stable_units_set)]
        stable_indices =  [list(stable_set) for stable_set in stable_units_set]
    
    
    stable_indices_new = fix_dead_stable_units(stable_indices_new, pruned_model)
        
    freeze_masks_new, drop_masks = compute_freeze_and_drop(stable_indices, pruned_model)
    pruned_model, num_dropped_connections  = drop_connections(pruned_model, drop_masks, DEVICE)
    
    return pruned_model, freeze_masks_new, num_dropped_connections, stable_indices, stable_units_set

#Find the percentage of frozen connections
def connection_freeze_perc(freeze_masks, prune_perc):
    layers = []
    for freeze_mask in freeze_masks:
        a = freeze_mask[0].shape
        total_connections = int(np.prod(a)*(1 - prune_perc / 100))
        frozen_connections =  np.sum(freeze_mask[0])
        layers.append(100*(frozen_connections / total_connections))
    return layers


#Fix dead plastic units
def fix_no_outgoing(model, plastic_indices, num_dropped_connections):
    weight = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            weight.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy()))
    
    new_masks = [weight[0][0]]
    bias_masks = [weight[0][1]]
    for i, layer_indices in enumerate(plastic_indices[1:-1], start = 1):
        M = weight[i][0]
        kernel_size = (M.shape[2] * M.shape[3]) if len(M.shape) == 4 else 1
        for unit_index in layer_indices:
            #not enough connections to fix all
            if num_dropped_connections[i] == 0 or len(plastic_indices[i + 1]) == 0:
                continue
            #Dead unit
            if np.sum(np.abs(M[:, unit_index])) == 0:
                target = np.random.choice(plastic_indices[i + 1], 1)
                M[target, unit_index] = np.ones((M.shape[2], M.shape[3])) if len(M.shape) == 4 else 1
                num_dropped_connections[i] =  num_dropped_connections[i] - kernel_size
        new_masks.append(M)
        bias_masks.append(weight[i][1])
    new_masks.append(weight[-1][0])
    bias_masks.append(weight[-1][1])
    model.set_masks(new_masks, bias_masks)
    return model, num_dropped_connections
        
                
#Degrade dead stable units to plastic
def fix_dead_stable_units(stable_indices,  model):
    weight = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            weight.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy()))
            
    new_stable_units = [stable_indices[0]]
    for i, (source_stable, target_stable) in enumerate(zip(stable_indices[:-1], stable_indices[1:])):
        source_stable, target_stable = np.array(source_stable, dtype=np.int32), np.array(target_stable, dtype=np.int32)
        dead_stable_units = []
        for tgt in target_stable:
            if np.sum(np.abs(weight[i][0][tgt, source_stable])) == 0:
                dead_stable_units.append(tgt)
        if len(dead_stable_units) != 0:
            updated_stable_indices = list(target_stable)
            for dead_stable in dead_stable_units:
                updated_stable_indices.remove(dead_stable)
            new_stable_units.append(updated_stable_indices)
        else:
            new_stable_units.append(list(target_stable))
    return new_stable_units

#Get list of weights
def list_of_weights(network):
    weights = []
    for module in network.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            weights.append((module.weight.cpu().detach().numpy(), module.bias.cpu().detach().numpy()))
    return weights

#Compute total activation of layers
def compute_total_activation(network, train_loader, dev, current_classes = None, binary = False):
    total_activations = []
    network.train()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(train_loader):
            _, activations = network.forward_activations(data, current_classes)
            batch_sum_activation = [torch.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else  torch.sum(activation, axis = 0) for activation in activations]
            
            if len(total_activations) == 0:
                total_activations = batch_sum_activation
            else:
                total_activations = [total_activations[i]+activation for i, activation in enumerate(batch_sum_activation)]

    return [total_activation.detach().cpu().numpy() for total_activation in total_activations], list_of_weights(network)

#Compute total average activation of layers (this is effectively same with above, only difference is normalization)
def compute_average_activation(network, train_loader, dev, current_classes = None, binary = False):
    total_activations = []
    network.train()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(train_loader):
            _, activations = network.forward_activations(data, current_classes)
            batch_sum_activation = [torch.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else  torch.sum(activation, axis = 0) for activation in activations]
            
            if len(total_activations) == 0:
                total_activations = batch_sum_activation
            else:
                total_activations = [total_activations[i]+activation for i, activation in enumerate(batch_sum_activation)]
                
    average_activations = [total_activation/len(train_loader.dataset) for total_activation in total_activations]
    return [average_activation.detach().cpu().numpy() for average_activation in average_activations], list_of_weights(network)

# Pick most active units 
def pick_top_neurons(activations, percentage):
    total = sum(activations)
    accumulate = 0
    indices = []
    sort_indices = np.argsort(-activations)
    for index in sort_indices:
       accumulate += activations[index]
       indices.append(index)
       if accumulate >= total * percentage / 100:
           break
    return indices

#Find the stable units that satisfies p criterion
def compute_stable_neurons(activations, in_channels, stable_outputs, activation_perc, model_type): 
    stable_indices = []
    
    for layer_activation in activations[0 if model_type == "conv" or model_type == "conv2"  else 1:-1]:
        stable_indices.append(pick_top_neurons(layer_activation, activation_perc))
    return [list(range(in_channels))] + stable_indices + [stable_outputs]

#Compute which connections to freeze and drop
def compute_freeze_and_drop(stable_indices, model):
    weight = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            weight.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy()))
            
    #Create Freeze Masks        
    freeze_masks = []
    for i, (source_stable, target_stable) in enumerate(zip(stable_indices[:-1], stable_indices[1:])):
        source_stable, target_stable = np.array(source_stable, dtype=np.int32), np.array(target_stable, dtype=np.int32)
        mask_w = np.zeros(weight[i][0].shape)
        #Conv2Conv
        if len(weight[i][0].shape) == 4:
            for src_unit_stable in source_stable:
               mask_w[target_stable, src_unit_stable, :, :] = np.ones((weight[i][0].shape[2], weight[i][0].shape[3]))
        #Conv2Linear or Linear2Linear
        else:
            #Conv2Linear
            if len(weight[i-1][0].shape) == 4:
                for src_unit_stable in source_stable:
                    
                    mask_w[target_stable, src_unit_stable*model.conv2lin_kernel_size:(src_unit_stable + 1)*model.conv2lin_kernel_size] = 1 
            #Linear2Linear
            else:
                for src_unit_stable in source_stable:
                    mask_w[target_stable, src_unit_stable] = 1 
        mask_b = np.zeros(weight[i][1].shape)
        mask_b[target_stable] = 1
        freeze_masks.append((mask_w, mask_b))
                
    #Create Drop Masks
    drop_masks = []
    for i, (mask_w, mask_b) in enumerate(freeze_masks):
        mask_w_drop = np.zeros(weight[i][0].shape)
        #Conv2Conv
        if len(weight[i][0].shape) == 4:
            for unit_index, b in enumerate(mask_b):
                if b:
                    mask_w_drop[unit_index, :, :, :] = np.logical_not(mask_w[unit_index])
            
        #Conv2Linear or Linear2Linear
        else:
            mask_w_drop = np.zeros(weight[i][0].shape)
            for unit_index, b in enumerate(mask_b):
                if b:
                    mask_w_drop[unit_index, :] = np.logical_not(mask_w[unit_index, :])
        drop_masks.append((mask_w_drop, []))
    return freeze_masks, drop_masks

# Calculate the sparsity of the model
def compute_weight_sparsity(model):
    parameters = 0
    ones = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            shape = module.weight.data.shape
            parameters += torch.prod(torch.tensor(shape))
            w_mask, _ = copy.deepcopy(module.get_mask())
            ones += torch.count_nonzero(w_mask)
    return float((parameters - ones) / parameters) * 100

# Drop connections plastic -> stable
def drop_connections(model, drop_masks, device):
    #Drop connections
    mask_index = 0
    num_drops = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            before = torch.sum(weight_mask)
            weight_mask[torch.tensor(drop_masks[mask_index][0], dtype= bool)] = 0 
            after = torch.sum(weight_mask)
            num_drops.append(int(before - after))
            module.set_mask(weight_mask, bias_mask)
            mask_index += 1
    return model, num_drops
