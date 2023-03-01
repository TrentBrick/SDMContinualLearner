from .config import DEVICE
from .replay_utils import Memory
from torch.utils.data import DataLoader
from .model import MLP,  ConvNet, random_prune

from .utils import (compute_average_activation, compute_freeze_and_drop,
                         compute_weight_sparsity, fix_no_outgoing, list_of_weights,
                         cosine_anneling, linear, exp_decay, get_masks, connection_freeze_perc, compute_plastic,
                         compute_stable_and_drop_connections, taskset2GPUdataset)

from.report import task_summary_log
import torch
from torch import nn, optim
import random
import numpy as np
from .training import training_CL, training_CL_memory
import copy
from .grow import grow_connection_A2B, connect_output2plastics
from continuum.tasks import split_train_val
from continuum import rehearsal
from torch.utils.data.dataloader import default_collate


class Learner():
    def __init__(self, args, input_dim, output_dim, scenario_train, scenario_test, deterministic = True, train_tranform = None, test_transform = None): 
        self.args = args
        if args.optimizer == "adam":
            self.optim_obj = optim.Adam
        elif args.optimizer == "ada_delta":
            self.optim_obj= optim.Adadelta
        else:
            self.optim_obj= optim.SGD
    
        if deterministic:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

        self.train_transform = train_tranform
        self.test_transform = test_transform

        self.Memo_train = Memory(scenario_train, args.memo_size, scenario_train.nb_classes)
            
        self.CIL = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        if args.model == 'conv':
            self.model = ConvNet(input_dim, output_dim).to(DEVICE)
        else:
            self.model = MLP(input_dim, output_dim).to(DEVICE)
        print("Model: \n", self.model)
        
        self.scenario_train = scenario_train
        self.scenario_test = scenario_test
        self.pruned_model = random_prune(self.model, args.prune_perc)
        
        self.freeze_masks = None
        self.freeze_masks_old = None
        self.drop_masks = None
        self.stable_indices = None
        self.num_dropped_connections = 0
        self.classes_so_far = []
        self.stable_units_set = []
        
        if self.args.p_step_size == 'cosine':
            self.p_schedule = lambda t: cosine_anneling(t, self.args.step_size_param)
        elif self.args.p_step_size == 'exp_decay':
            self.p_schedule = lambda t: exp_decay(t, self.args.step_size_param)
        else: 
            self.p_schedule = lambda t: linear(t, self.args.step_size_param)
            
    #task_id should start from 0
    #taskset is object of continuum library 
    def task_train(self, task_id, taskset, val_set = None):
        current_classes = list(taskset.get_classes())
        self.classes_so_far += current_classes
        print("Task ID: ", task_id + 1, " Classes: ", current_classes)
        #Create data loaders
        if val_set is None:
            train_taskset, val_taskset = split_train_val(taskset, val_split=0.1)
            val_loader = DataLoader(val_taskset,  batch_size= self.args.batch_size,  shuffle=True, collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        else:
            train_taskset = taskset
            val_loader = DataLoader(val_set,  batch_size= self.args.batch_size,  shuffle=True, collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        

        train_loader = DataLoader(train_taskset,  batch_size= self.args.batch_size,  shuffle=True, collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        
        #Get training objects
        loss = nn.CrossEntropyLoss()
        
        #Iterative FineTuning Step
        previous_model = None
        current_model = self.pruned_model
        stable_units_set =  self.stable_units_set
        stable_indices = self.stable_indices
        stable_indices_prev = None
        p_prev = 100
        p_new = 100
        acc_prev = 0
        connectivity_before = copy.deepcopy(get_masks(self.pruned_model))
        print('Starting:')
        t = 0
        list_of_ps = []
        num_stable_units = []
        
        while(True):
            t = t + 1
            stable_perc = self.p_schedule(t) * 100
            
            print('Model sparsity before training:', compute_weight_sparsity(current_model))

            #Reset optimizer for every task
            optimizer = self.optim_obj(current_model.parameters(), lr= self.args.learning_rate, weight_decay= 0)
            connectivity_before = copy.deepcopy(get_masks(current_model))
           
            try:
                memo_loader = DataLoader(self.Memo_train.create_dataset(), batch_size = self.args.batch_size, shuffle=True,
                collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
            except:
                memo_loader = None
            current_model, _, _, acc_after_fine_tuning = training_CL_memory(
                current_model, loss, optimizer, train_loader,
                memo_loader, val_loader, self.args, 
                self.args.phase_epochs, freeze_masks = self.freeze_masks)
                
            acc_new = acc_after_fine_tuning[-1]
            #Accuracy Dropped stop iteration
            if (acc_new < (acc_prev - self.args.recovery_perc) and t > self.args.min_phases) or stable_perc <= 5:
                self.pruned_model = previous_model
                if len(self.stable_units_set) == 0:
                    self.stable_units_set = [set(stable_index) for stable_index in stable_indices_prev]
                    self.stable_indices =  [list(stable_set) for stable_set in self.stable_units_set]
                else:
                    self.stable_units_set = [prev_set.union(set(stable_index)) for stable_index, prev_set in zip(stable_indices_prev, self.stable_units_set)]
                    self.stable_indices =  [list(stable_set) for stable_set in self.stable_units_set]
                self.freeze_masks, _ = compute_freeze_and_drop(self.stable_indices, self.pruned_model)
                if  self.args.reinit:
                    self.pruned_model.re_initialize(self.freeze_masks)
                break
            else:
                acc_prev = max(acc_new, acc_prev)
                previous_model = current_model
                stable_indices_prev = stable_indices
                p_prev = p_new
                num_stable_units.append(stable_indices_prev)
                list_of_ps.append(p_prev)
            
            print('Dropping for p:{}:'.format(stable_perc))
            
            current_model, _, num_dropped_connections, stable_indices, stable_units_set = compute_stable_and_drop_connections(
                                                                               current_model,  self.input_dim, self.stable_units_set, 
                                                                               train_loader, current_classes, 
                                                                               self.classes_so_far, stable_perc,  model_type= self.args.model )

            if  self.args.grow:
                #Grow connections
                connectivity_after = copy.deepcopy(get_masks(current_model))
                num_dropped_connections =  [torch.sum(torch.abs(a  - b)).cpu().numpy() for a, b in zip(connectivity_before, connectivity_after)]
                plastic_units = compute_plastic(current_model, stable_indices)
                #Fix Non importants without outgoing connections
                current_model, remainder_connections = fix_no_outgoing(current_model, plastic_units, num_dropped_connections)
                weights = list_of_weights(current_model)
                

                activations, _ = compute_average_activation(current_model, train_loader, DEVICE, current_classes, False)
                current_model, remainder_connections = grow_connection_A2B(current_model, remainder_connections,
                                                       [weights[0][0].shape[1]]  + [w.shape[0] for w, _ in weights], 
                                                       stable_indices,  activations, 
                                                       grow_algo = self.args.rewire_algo, 
                                                       weight_init_algo = self.args.grow_init)
              
                #Alternative growth
                current_model, remainder_connections = grow_connection_A2B(current_model, remainder_connections,
                                                       [weights[0][0].shape[1]]  + [w.shape[0] for w, _ in weights], 
                                                       stable_indices,  activations, 
                                                       grow_algo = 'full_random', 
                                                       weight_init_algo = self.args.grow_init)
            
            p_new = stable_perc
        print('End:')
        
        self.Memo_train.push_samples(task_id)

        percentange_of_frozen_conns = connection_freeze_perc(self.freeze_masks, self.args.prune_perc)
        
        backward_accs, backward_tasks, cum_acc = task_summary_log(self.pruned_model, loss,
                                current_classes, self.args, 
                                task_id, self.scenario_test, self.stable_units_set)

        
        

        return p_prev, percentange_of_frozen_conns, backward_accs[0], 0, list_of_ps[1:], num_stable_units[1:]

