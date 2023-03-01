import copy
import torch
import torch.nn as nn
from .config import DEVICE


def reset_frozen_gradients(network, freeze_masks):
    mask_index = 0
    for module in network.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[freeze_masks[mask_index][0]] = 0
            module.bias.grad[freeze_masks[mask_index][1]] = 0
            mask_index += 1
    return network



#Accepts number of epochs as a parameter
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
            #import ipdb 
            #ipdb.set_trace()

            optimizer.zero_grad()
            output = network(data, current_classes)
                
            batch_loss = loss(output, target.long())
            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            if freeze_masks is not None:
                network = reset_frozen_gradients(network, freeze_masks)
            optimizer.step()
            
        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = current_classes if multihead else None, report=False)
        accuracy1.append(acc1)
        test_loss.append(avg_loss)

        if acc1 > test_acc1_min:
            best_network = copy.deepcopy(network)
            best_epoch = epoch
            test_acc1_min = acc1
        
    return best_network, train_curve[:best_epoch+1], test_loss[:best_epoch+1], accuracy1[:best_epoch+1]

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