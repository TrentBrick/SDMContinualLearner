import copy
import torch
import torch.nn as nn
from .config import DEVICE


class Cycle:
    def __init__(self, loader):
        self.loader = loader
        self.generator = iter(loader)

    def get_batch(self):
        try:
            # Samples the batch
            return next(self.generator)
        except:
            # restart the generator if the previous generator is exhausted.
            self.generator = iter(self.loader)
            return next(self.generator)

#freeze_degree:  'full_freeze', "except_final", "slow_update" "no_freeze"
def reset_frozen_gradients(network, freeze_masks):
    mask_index = 0
    l = list(network.modules())[1:-1]

    for module in l:
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
            module.weight.grad[freeze_masks[mask_index][0]] = 0
            module.bias.grad[freeze_masks[mask_index][1]] = 0
            mask_index += 1
    return network



def training_CL_memory(network, loss, optimizer, train_loader, memo_loader, test_loader,  args, epochs, freeze_masks = None):
    if memo_loader is not None:
        memo_loader_cycle = Cycle(memo_loader)  

    if freeze_masks is not None:
        freeze_masks = [(torch.tensor(w).to(torch.bool).to(DEVICE) , torch.tensor(b).to(torch.bool).to(DEVICE)) for w, b in freeze_masks]
    #Training
    train_curve = []
    accuracy1 = []
    test_loss = []

    for epoch in range(epochs):
        network.train()
        train_loss = 0
        for batch_idx, (data, target, _) in enumerate(train_loader):
            #print(torch.unique(target, return_counts=True))
            optimizer.zero_grad()
            output = network(data)
                
            if memo_loader is not None:
                data_memo, target_memo =  memo_loader_cycle.get_batch()
                output_memo = network(data_memo.to(DEVICE))
                l1 = loss(output, target.long())
                l2 = loss(output_memo, target_memo.to(DEVICE).long())
                batch_loss = l1  + args.replay_lambda *  l2
            else:
                batch_loss = loss(output, target.long())

            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            if freeze_masks is not None:
                network = reset_frozen_gradients(network, freeze_masks)
            optimizer.step()
            
        train_curve.append(train_loss/len(train_loader.dataset))
        avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = None, report=True)
            
        accuracy1.append(acc1)
        test_loss.append(avg_loss)

        
    return copy.deepcopy(network), train_curve, test_loss, accuracy1




#Accepts number of epochs as a parameter
def training_CL(network, loss, optimizer, train_loader, test_loader,  args, epochs,
                    current_classes = None, freeze_masks = None, mask_outputs = False): 
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
        for batch_idx, (data, target, _) in enumerate(train_loader):

            #print(torch.unique(target, return_counts=True))
            optimizer.zero_grad()
            #Mask out non task output units if mask_outputs == True
            if mask_outputs:
                output = network(data, current_classes)
            else:
                output = network(data)
                
            try:
                batch_loss = loss(output, target.long())
            except:
                batch_loss =  loss(output, nn.functional.one_hot(target, num_classes=10).float()) 

            train_loss += batch_loss.item() * data.size(0)
            batch_loss.backward()
            if freeze_masks is not None:
                network = reset_frozen_gradients(network, freeze_masks, args.no_freeze_last)
            optimizer.step()
            
        train_curve.append(train_loss/len(train_loader.dataset))
        #Mask unrelated outputs during testing
        if mask_outputs:
            avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = current_classes, report=True)
        else:
            avg_loss, acc1, _ = test(network, loss, test_loader, current_classes = None, report=True)
            
        accuracy1.append(acc1)
        test_loss.append(avg_loss)

        if acc1 > test_acc1_min:
            best_network = copy.deepcopy(network)
            best_epoch = epoch
            test_acc1_min = acc1
        
    return best_network, train_curve[:best_epoch+1], test_loss[:best_epoch+1], accuracy1[:best_epoch+1]

def test(network, loss, dataloader,  title = "test", current_classes = None, report = False):
    network.eval()
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for idx, (data, target, _) in enumerate(dataloader):
            output = network(data, current_classes)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1,1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()

    acc1 = 100.0 * correct1 / len(dataloader.dataset)
    acc5 = 100.0 * correct5 / len(dataloader.dataset)

    if report:
        print('[{}] Top 1 Accuracy ='.format(title), acc1)

    return None, acc1, acc5