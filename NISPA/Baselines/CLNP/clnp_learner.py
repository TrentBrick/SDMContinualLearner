import models
import utils
from config import DEVICE
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import copy
from continuum.tasks import split_train_val


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


def accuracy_on_tasks_so_far(model, loss, current_classes, task_id, scenario_test):
    accs = []
    for backward_task_id in range(task_id + 1):
        backward_task_dataset = scenario_test[backward_task_id]
        test_loader = DataLoader(backward_task_dataset, batch_size= 512, shuffle=False)
        test_acc = utils.test(model, loss, test_loader, current_classes = backward_task_dataset.get_classes())
        accs.append((backward_task_id, test_acc))
    return accs

class CL_Learner_Conv():
    def __init__(self, args, input_dim, output_dim, scenario_train, scenario_test): 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = models.ConvNet(input_dim, output_dim).to(DEVICE)
        print("Model: \n", self.model)
        self.args = args
        self.scenario_train = scenario_train
        self.scenario_test = scenario_test
        self.freeze_masks = None
        self.classes_so_far = []
        self.important_units = []
        
    def task_train(self, task_id, taskset):
        current_classes = list(taskset.get_classes())
        self.classes_so_far += current_classes
        print("Task ID: ", task_id + 1, " Classes: ", current_classes)
        #Create data loaders
        train_taskset, val_taskset = split_train_val(taskset, val_split=0.1)
        train_loader = DataLoader(train_taskset,  batch_size= self.args.batch_size,  shuffle=True)
        val_loader = DataLoader(val_taskset,  batch_size= self.args.batch_size,  shuffle=True)
        
        test_set = self.scenario_test[task_id]  
        test_loader = DataLoader(test_set, batch_size = self.args.batch_size*2, shuffle=False)
        #Get training objects
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr= self.args.learning_rate, weight_decay= 0)
        print('Model sparsity before training:', compute_weight_sparsity(self.model))
        trained_model = utils.train(self.model, loss, optimizer, train_loader, val_loader,
                                    self.args, current_classes, freeze_masks = self.freeze_masks, epochs = self.args.epochs)
        
        best_acc = utils.test(trained_model, loss, val_loader, current_classes)
        worst_acc = best_acc - (self.args.m_perc)
        _, upper = utils.get_activations(trained_model, train_loader, current_classes)
        prev_model = None
        prev_freeze_mask = None
        prev_important_units = self.important_units
        print("Start Pruning")
        for theta in np.arange(0, upper, self.args.theta_step):
            important_units = utils.get_important_units(trained_model, train_loader, theta, self.important_units, self.input_dim,
                                                        self.classes_so_far, current_classes, conv_input = True)
            

                   
            pruned_model, freeze_mask = utils.prune(copy.deepcopy(trained_model), important_units)
            pruned_acc = utils.test(pruned_model, loss, val_loader, current_classes)

            #No new unit needed
            if all([len(set(new).difference(set(old))) == 0 for old, new in zip(self.important_units[:-1], important_units[:-1])])  and  len(self.important_units) != 0:
                print("No new unit")
                self.model = pruned_model
                self.freeze_masks = freeze_mask
                self.important_units = important_units
                print([len(units) for units in self.important_units])
                break
            
            #We can further prune
            if pruned_acc >= worst_acc:
                #No new unit needed
                if theta == upper:
                    print("No new unit needed")
                    self.model = pruned_model
                    self.freeze_masks = freeze_mask
                    self.important_units = important_units
                    break
                prev_model = copy.deepcopy(pruned_model)
                prev_freeze_mask = freeze_mask
                prev_important_units = important_units
            #Stop pruning and revert
            else:
                self.model = copy.deepcopy(prev_model)
                self.freeze_masks = prev_freeze_mask
                self.important_units = prev_important_units
                break
        self.model.re_initialize(self.freeze_masks)
        print("Task theta: ", theta)
        task_acc = utils.test(self.model, loss, test_loader, current_classes)
        print("TASK {} END: ".format(current_classes), task_acc)
        return accuracy_on_tasks_so_far(self.model, loss, current_classes, task_id, self.scenario_test)
        
        
        
class CL_Learner_MLP():
    def __init__(self, args, input_dim, output_dim, scenario_train, scenario_test): 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = models.MLP(input_dim, output_dim).to(DEVICE)
        print("Model: \n", self.model)
        self.args = args
        self.scenario_train = scenario_train
        self.scenario_test = scenario_test
        self.freeze_masks = None
        self.classes_so_far = []
        self.important_units = []
        
    def task_train(self, task_id, taskset):
        current_classes = list(taskset.get_classes())
        self.classes_so_far += current_classes
        print("Task ID: ", task_id + 1, " Classes: ", current_classes)
        #Create data loaders
        train_taskset, val_taskset = split_train_val(taskset, val_split=0.1)
        train_loader = DataLoader(train_taskset,  batch_size= self.args.batch_size,  shuffle=True)
        val_loader = DataLoader(val_taskset,  batch_size= self.args.batch_size,  shuffle=True)
        
        test_set = self.scenario_test[task_id]  
        test_loader = DataLoader(test_set, batch_size = self.args.batch_size*2, shuffle=False)
        #Get training objects
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr= self.args.learning_rate, weight_decay= 0)
        print('Model sparsity before training:', compute_weight_sparsity(self.model))
        trained_model = utils.train(self.model, loss, optimizer, train_loader, val_loader,
                                    self.args, current_classes, freeze_masks = self.freeze_masks, epochs = self.args.epochs)
        
        best_acc = utils.test(trained_model, loss, val_loader, current_classes)
        worst_acc = best_acc - (self.args.m_perc)
        _, upper = utils.get_activations(trained_model, train_loader, current_classes)
        prev_model = None
        prev_freeze_mask = None
        prev_important_units = self.important_units
        #Prune iteratively
        print("Start Pruning")
        for theta in np.arange(0, upper, self.args.theta_step):
            important_units = utils.get_important_units(trained_model, train_loader, theta, self.important_units, self.input_dim,
                                                        self.classes_so_far, current_classes, conv_input = False)
            
            pruned_model, freeze_mask = utils.prune(copy.deepcopy(trained_model), important_units)
            pruned_acc = utils.test(pruned_model, loss, val_loader, current_classes)

            #No new unit needed
            if all([len(set(new).difference(set(old))) == 0 for old, new in zip(self.important_units[:-1], important_units[:-1])])  and  len(self.important_units) != 0:
                print("No new unit")
                self.model = pruned_model
                self.freeze_masks = freeze_mask
                self.important_units = important_units
                print([len(units) for units in self.important_units])
                break
            
            #We can further prune
            if pruned_acc >= worst_acc:
                #No new unit needed
                if theta == upper:
                    print("No new unit needed")
                    self.model = pruned_model
                    self.freeze_masks = freeze_mask
                    self.important_units = important_units
                    break
                prev_model = copy.deepcopy(pruned_model)
                prev_freeze_mask = freeze_mask
                prev_important_units = important_units
            #Stop pruning and revert
            else:
                self.model = copy.deepcopy(prev_model)
                self.freeze_masks = prev_freeze_mask
                self.important_units = prev_important_units
                break
        self.model.re_initialize(self.freeze_masks)
        print("Task theta: ", theta)
        task_acc = utils.test(self.model, loss, test_loader, current_classes)
        print("TASK {} END: ".format(current_classes), task_acc)
        return accuracy_on_tasks_so_far(self.model, loss, current_classes, task_id, self.scenario_test)