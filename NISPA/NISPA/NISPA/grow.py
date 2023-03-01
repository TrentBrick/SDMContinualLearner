import numpy as np
import torch
import copy
from .config import DEVICE


    
# Grow randomly between plastic -> plastic and stable -> plastic
def full_random(model, all_units, stable_indices):
    w_masks = get_w_masks(model)
    pos_connections = []
    bias_matrices = []
    layer_id = 0
    for (stable_src, stable_tgt), (all_unit_src, all_unit_tgt) in zip( zip(stable_indices[:-1], stable_indices[1:]), zip (all_units[:-1], all_units[1:])):
        plastic_tgt = list(set(list(range(all_unit_tgt))).difference(stable_tgt)) 
        conn_type_1 = np.ones((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 1
        conn_type_0 = np.zeros((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 0
        pos_conn = np.zeros(w_masks[layer_id].shape)
        pos_conn[plastic_tgt,:] = conn_type_1
        if len(w_masks[layer_id].shape) == 4:
            pos_conn[np.all(w_masks[layer_id][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
        else:
            pos_conn[w_masks[layer_id] != 0] = 0
        if len(w_masks[layer_id].shape) == 4:
            bias_matrix = copy.deepcopy(pos_conn.sum(axis = (2, 3)))
        else:
            bias_matrix = copy.deepcopy(pos_conn)
        bias_matrix[np.nonzero(bias_matrix)] =  bias_matrix[np.nonzero(bias_matrix)] / np.sum(bias_matrix)
        pos_connections.append(pos_conn)
        bias_matrices.append(bias_matrix)
        layer_id = layer_id + 1
    return pos_connections, bias_matrices


# Grow randomly between stable -> plastic
def forwardT_random(model, all_units, stable_indices):
    w_masks = get_w_masks(model)
    pos_connections = []
    bias_matrices = []
    layer_id = 0
    for (stable_src, stable_tgt), (all_unit_src, all_unit_tgt) in zip( zip(stable_indices[:-1], stable_indices[1:]), zip (all_units[:-1], all_units[1:])):
        plastic_tgt = list(set(list(range(all_unit_tgt))).difference(stable_tgt)) 
        conn_type_1 = np.ones((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 1
        conn_type_0 = np.zeros((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 0
        pos_conn = np.zeros(w_masks[layer_id].shape)
        pos_conn[np.ix_(plastic_tgt,stable_src)] = conn_type_1
        if len(w_masks[layer_id].shape) == 4:
            pos_conn[np.all(w_masks[layer_id][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
        else:
            pos_conn[w_masks[layer_id] != 0] = 0
        if len(w_masks[layer_id].shape) == 4:
            bias_matrix = copy.deepcopy(pos_conn.sum(axis = (2, 3)))
        else:
            bias_matrix = copy.deepcopy(pos_conn)
        bias_matrix[np.nonzero(bias_matrix)] =  bias_matrix[np.nonzero(bias_matrix)] / np.sum(bias_matrix)
        layer_id = layer_id + 1
        pos_connections.append(pos_conn)
        bias_matrices.append(bias_matrix)
    return pos_connections, bias_matrices

# Grow randomly between plastic -> plastic
def flow_random(model, all_units, stable_indices):
    w_masks = get_w_masks(model)
    pos_connections = []
    bias_matrices = []
    layer_id = 0
    for (important_src, important_tgt), (all_unit_src, all_unit_tgt) in zip( zip(important_indices[:-1], important_indices[1:]), zip (all_units[:-1], all_units[1:])):
        unimportant_tgt = list(set(list(range(all_unit_tgt))).difference(important_tgt)) 
        unimportant_src = list(set(list(range(all_unit_src))).difference(important_src)) 
        conn_type_1 = np.ones((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 1
        conn_type_0 = np.zeros((w_masks[layer_id].shape[2], w_masks[layer_id].shape[3])) if len(w_masks[layer_id].shape) == 4 else 0
        pos_conn = np.zeros(w_masks[layer_id].shape)
        pos_conn[np.ix_(unimportant_tgt,unimportant_src)] = conn_type_1
        if len(w_masks[layer_id].shape) == 4:
            pos_conn[np.all(w_masks[layer_id][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
        else:
            pos_conn[w_masks[layer_id] != 0] = 0
        if len(w_masks[layer_id].shape) == 4:
            bias_matrix = copy.deepcopy(pos_conn.sum(axis = (2, 3)))
        else:
            bias_matrix = copy.deepcopy(pos_conn)
        bias_matrix[np.nonzero(bias_matrix)] =  bias_matrix[np.nonzero(bias_matrix)] / np.sum(bias_matrix)
        layer_id = layer_id + 1
        pos_connections.append(pos_conn)
        bias_matrices.append(bias_matrix)
    return pos_connections, bias_matrices
        
        
# Different initilizations for newly added connections
def weight_init(module, weight_init_algo, size):
    if weight_init_algo == 'normal':
        w =  module.weight.data.cpu().numpy()
        non_zero_weights =  w[np.nonzero(w)] 
        weights= np.random.normal(float(np.mean(non_zero_weights)), float(np.std(non_zero_weights)), size = size)
    elif weight_init_algo == 'uniform':  
            weights = np.random.uniform(float(torch.min(module.weight.data)), float(torch.max(module.weight.data)), size = size)
    elif weight_init_algo == 'zero':
            weights = np.zeros(size) 
    else:
        raise Exception('Undefined weight init algorithm: {}'.format(weight_init_algo)) 
    return weights
        

def get_w_masks(model):
    w_masks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            w_masks.append(copy.deepcopy(weight_mask).cpu().numpy())
    return w_masks


def get_possible_connections(model, all_units, stable_indices,   activations,  grow_algo = 'forwardT_most_active'):
    if grow_algo == 'full_random':
        return full_random(model, all_units, stable_indices)
    elif  grow_algo =='forwardT_random':
        return forwardT_random(model, all_units, stable_indices)
    elif  grow_algo =='flow_random':
        return flow_random(model, all_units, stable_indices)
    else:
        raise Exception('Undefined growth algorithm: {}'.format(grow_algo)) 


def grow_connection_A2B(model, num_connections, all_units, stable_indices, activations,  grow_algo = 'forwardT_most_active',  weight_init_algo = 'zero'):
    #Sanity Check
    if sum(num_connections) == 0:
        return model, num_connections
    possible_connections, bias_matrix = get_possible_connections(model, all_units, stable_indices, activations, grow_algo)
    layer_id = 0 
    remainder_connections = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            if num_connections[layer_id] == 0:
                remainder_connections.append(0)
                layer_id = layer_id + 1 
                continue
            weight_mask, bias_mask = module.get_mask()
            #Conv layer
            if len(possible_connections[layer_id].shape) == 4:
                grow_indices = np.nonzero(np.sum(possible_connections[layer_id], axis = (2 , 3)))
                init_weights = weight_init(module, weight_init_algo, (len(grow_indices[0]), possible_connections[layer_id].shape[2], possible_connections[layer_id].shape[3]))
            #Linear Layer
            else:
                grow_indices = np.nonzero(possible_connections[layer_id])
                init_weights = weight_init(module, weight_init_algo, len(grow_indices[0]))
            conn_type = np.ones((possible_connections[layer_id].shape[2], possible_connections[layer_id].shape[3])) if len(possible_connections[layer_id].shape) == 4 else 1
            conn_size = (possible_connections[layer_id].shape[2] * possible_connections[layer_id].shape[3]) if len(possible_connections[layer_id].shape) == 4 else 1
            
            probs = bias_matrix[layer_id][grow_indices]
            
            
            
            #There are connections that we can grow
            if len(grow_indices[0]) != 0:
                #We can partial accommodate grow request (we will have remainder connections)
                if len(grow_indices[0])*conn_size <= num_connections[layer_id]:
                    weight_mask[grow_indices] = torch.tensor(conn_type, dtype = weight_mask.dtype).to(DEVICE)
                    module.weight.data[grow_indices] = torch.tensor(init_weights, dtype = torch.float32).to(DEVICE)
                    remainder_connections.append(num_connections[layer_id] - len(grow_indices[0])*conn_size)
                else:
                    #Biased or random selection from possible connections
                    try:
                        selection = np.random.choice(len(grow_indices[0]), size = int(num_connections[layer_id]/ conn_size), replace = False, p = probs)
                    except:
                        selection = np.random.choice(len(grow_indices[0]), size = int(num_connections[layer_id]/ conn_size), replace = False)
                        print("Not enough possible connections to sample properly! Layer: ",layer_id)
                    weight_mask[(grow_indices[0][selection], grow_indices[1][selection])] = torch.tensor(conn_type, dtype = weight_mask.dtype).to(DEVICE)
                    module.weight.data[grow_indices] = torch.tensor(init_weights, dtype = torch.float32).to(DEVICE)
                    remainder_connections.append(0)
            else:
                remainder_connections.append(num_connections[layer_id])
            module.set_mask(weight_mask, bias_mask)
            layer_id += 1
    return model, remainder_connections