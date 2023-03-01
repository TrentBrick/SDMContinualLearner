from config import DEVICE
import torch.nn as nn
import torch
import copy
import numpy as np

def reset_frozen_gradients(model, freeze_masks):
    mask_index = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[torch.tensor(freeze_masks[mask_index][0]).to(torch.bool)] = 0
            module.bias.grad[torch.tensor(freeze_masks[mask_index][1]).to(torch.bool)] = 0
            mask_index += 1
    return model

def get_activations(model, train_loader, current_classes):
    total_activations = []
    model.train()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            _, activations = model.forward_activations(data, current_classes)
            batch_sum_activation = [torch.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else  torch.sum(activation, axis = 0) for activation in activations]
            
            if len(total_activations) == 0:
                total_activations = batch_sum_activation
            else:
                total_activations = [total_activations[i]+activation for i, activation in enumerate(batch_sum_activation)]
                
    average_activations = [total_activation/len(train_loader.dataset) for total_activation in total_activations]
    np_activations = [average_activation.detach().cpu().numpy() for average_activation in average_activations]
    return np_activations, max([np.max(acti) for acti in np_activations])


def get_drop_and_freeze_masks(model, important_indices):
    weight = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            weight.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy()))
            
    #Create Freeze Masks        
    freeze_masks = []
    for i, (source_blue, target_blue) in enumerate(zip(important_indices[:-1], important_indices[1:])):
        source_blue, target_blue = np.array(source_blue, dtype=np.int32), np.array(target_blue, dtype=np.int32)
        mask_w = np.zeros(weight[i][0].shape).astype(np.bool)
        #Conv2Conv
        if len(weight[i][0].shape) == 4:
            for src_unit_blue in source_blue:
               mask_w[target_blue, src_unit_blue, :, :] = np.ones((weight[i][0].shape[2], weight[i][0].shape[3]))
        #Conv2Linear or Linear2Linear
        else:
            #Conv2Linear
            if len(weight[i-1][0].shape) == 4:
                for src_unit_blue in source_blue:
                    mask_w[target_blue, src_unit_blue*64:(src_unit_blue + 1)*64] = 1 
            #Linear2Linear
            else:
                for src_unit_blue in source_blue:
                    mask_w[target_blue, src_unit_blue] = 1 
        mask_b = np.zeros(weight[i][1].shape).astype(np.bool)
        mask_b[target_blue] = 1
        freeze_masks.append((mask_w, mask_b))
                
    #Create Drop Masks
    drop_masks = []
    for i, (mask_w, mask_b) in enumerate(freeze_masks):
        mask_w_drop = np.zeros(weight[i][0].shape).astype(np.bool)
        #Conv2Conv
        if len(weight[i][0].shape) == 4:
            for unit_index, b in enumerate(mask_b):
                if b:
                    mask_w_drop[unit_index, :, :, :] = np.logical_not(mask_w[unit_index])
            
        #Conv2Linear or Linear2Linear
        else:
            mask_w_drop = np.zeros(weight[i][0].shape).astype(np.bool)
            for unit_index, b in enumerate(mask_b):
                if b:
                    mask_w_drop[unit_index, :] = np.logical_not(mask_w[unit_index, :])
        drop_masks.append((mask_w_drop, []))
    return freeze_masks, drop_masks


def drop_connections(model, drop_masks):
    #Drop connections
    mask_index = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, nn.Conv2d):
            weight_mask, bias_mask = module.get_mask()
            weight_mask[torch.tensor(drop_masks[mask_index][0], dtype= bool)] = 0 
            module.set_mask(weight_mask, bias_mask)
            mask_index += 1
    return model

def get_important_units(model, train_loader, theta, old_important_units, input_dim, important_output_units, current_classes, conv_input = True):
    selected_indices = []
    activations, _ = get_activations(model, train_loader, current_classes)
    for layer_activation in activations[0 if conv_input else 1:-1]:
        indices = np.nonzero(layer_activation > theta)[0]
        selected_indices.append(indices)
    selected_indices = [list(range(input_dim))] + selected_indices + [important_output_units]
    if len(old_important_units) != 0:
        important_indices = [list(sorted(list(set(new).union(set(old))))) for new, old in zip(selected_indices, old_important_units)]
    else:
        important_indices = selected_indices
    return important_indices
    
def prune(model, important_indices):
    freeze_masks, drop_masks = get_drop_and_freeze_masks(model, important_indices)
    pruned_model =  drop_connections(model, drop_masks)
    return pruned_model, freeze_masks 

def train(model, loss, optimizer, train_loader, test_loader, args, current_classes, freeze_masks, epochs):
    #Early Stopping
    test_acc_min = 0
    best_model = model
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data, current_classes)
            batch_loss = loss(output, target.long()) + (args.l1_alpha * sum([p.abs().sum() for p in model.parameters()]))
            batch_loss.backward()
            if freeze_masks is not None:
                model = reset_frozen_gradients(model, freeze_masks)
            optimizer.step()
        
        acc_test = test(model, loss, test_loader, current_classes)
        if acc_test > test_acc_min:
            best_model = copy.deepcopy(model)
            test_acc_min = acc_test
   
    return best_model

def test(model, loss, test_loader, current_classes):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for idx, (data, target, _ ) in enumerate(test_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data, current_classes)
            total += loss(output, target.long()).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    acc1 = 100.0 * correct1 / len(test_loader.dataset)
    print('[Test] Top 1 Accuracy =', acc1)

    return acc1