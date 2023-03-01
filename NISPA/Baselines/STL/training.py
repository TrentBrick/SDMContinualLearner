import copy
import torch
import torch.nn as nn
import numpy as np
from config import DEVICE

def pick_k_from_rows(w_mask, k):
    picked = []
    for row_index, row in enumerate(w_mask):
        nonzeros = row.nonzero()[0]
        if len(nonzeros) == 0:
            continue
        else:
            #If we do not have any other option we may pick same edge twice
            indices = np.random.choice(nonzeros, k, replace = True if len(nonzeros) < k else False)
            picked.append([(row_index, ind) for ind in indices])
    return sum(picked, [])

def freeze2growdrop(freeze_masks, model, current_classes):
    if freeze_masks is None:
        freeze_masks = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight_mask, bias_mask = module.get_mask()
                freeze_masks.append((np.zeros_like(weight_mask.cpu().numpy(), dtype=np.bool) , np.zeros_like(bias_mask.cpu().numpy(), dtype=np.bool)))
    
    no_grow_masks = []
    no_drop_masks = []
    
    w_masks = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight_mask, bias_mask = module.get_mask()
            w_masks.append(copy.deepcopy(weight_mask).cpu().numpy())
            
    for w_mask, (weight, bias) in zip(w_masks, freeze_masks):
        
        #Frozen connections and path preserving connections
        w = copy.deepcopy(w_mask)
        frozen_w = copy.deepcopy(weight)
        frozen_w[tuple(zip(*pick_k_from_rows(w, 5)))] = True
        no_drop_masks.append((frozen_w, copy.deepcopy(bias)))
    
        #Incoming to Blue units
        w = copy.deepcopy(weight)
        w[np.sum(w, axis = 1) > 0, :] = True
        no_grow_masks.append((w, copy.deepcopy(bias)))
    
    #Only consider current head for rewiring
    output = copy.deepcopy(w_masks[-1]).astype(np.bool)
    output[current_classes, :] = 0
    no_drop_masks[-1] = (output, no_drop_masks[-1][1])
    
    
    return no_grow_masks, no_drop_masks

def reset_frozen_gradients(network, freeze_masks):
    mask_index = 0
    for module in network.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[freeze_masks[mask_index][0]] = 0
            module.bias.grad[freeze_masks[mask_index][1]] = 0
            mask_index += 1
    return network


#No Dynamic Sparse Training
#Accepts number of epochs as a parameter
#Ideal for No grow only dropping/freezing sparsity experiments
def training_CL(network, loss, optimizer, train_loader, test_loader,  args, epochs,
                    current_classes = None, freeze_masks = None, multihead = False): 
    #Early Stopping
    test_acc1_min = 0
    best_epoch = 0
    best_network = network
    if freeze_masks is not None:
        freeze_masks = [(torch.tensor(w).to(torch.bool).to(DEVICE) , torch.tensor(b).to(torch.bool).to(DEVICE)) for w, b in freeze_masks]
    #Training
    train_curve = []
    accuracy1 = []
    test_loss = []
    for epoch in range(epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data, current_classes)
            batch_loss = loss(output, target.long())
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            if freeze_masks is not None:
                network = reset_frozen_gradients(network, freeze_masks)
            optimizer.step()
            
        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = current_classes if multihead else None, report=True)
        accuracy1.append(acc1)
        test_loss.append(avg_loss)

        if acc1 > test_acc1_min:
            best_network = copy.deepcopy(network)
            best_epoch = epoch
            test_acc1_min = acc1
        
    return best_network, train_curve[:best_epoch+1], test_loss[:best_epoch+1], accuracy1[:best_epoch+1]


def naive_training(network, loss, optimizer, train_loader, test_loader,  args, 
                    current_classes = None, multihead = False):
    
    #Early Stopping
    acc1_min = 0
    best_epoch = 0
    best_network = None

    
    train_curve = []
    accuracy1 = []
    test_loss = []
    for epoch in range(args.epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data, current_classes)
            batch_loss = loss(output, target.long())
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            optimizer.step()
 
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss.item()))

                
        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = current_classes if multihead else None)
        print("TEST ACC: ", acc1)
        accuracy1.append(acc1)
        test_loss.append(avg_loss)
        
            
        if acc1 > acc1_min:
            best_network = copy.deepcopy(network)
            best_epoch = epoch
            acc1_min = acc1

    return best_network, train_curve[:best_epoch+1], test_loss[:best_epoch+1], accuracy1[best_epoch]
    
    


def test(network, loss, dataloader,  title = "test", current_classes = None, report = False):
    network.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            output = network(data, current_classes)
            total += loss(output, target.long()).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    avg_loss = total / len(dataloader.dataset)
    acc1 = 100.0 * correct1 / len(dataloader.dataset)
    acc5 = 100.0 * correct5 / len(dataloader.dataset)

    if report:
        print('[{}] Top 1 Accuracy ='.format(title), acc1)
        print('[{}] Top 5 Accuracy ='.format(title), acc5)
        print('[{}] Average Loss ='.format(title), avg_loss)

    return avg_loss, acc1, acc5