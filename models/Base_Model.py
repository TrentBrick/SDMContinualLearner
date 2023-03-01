import copy
import pickle
from textwrap import indent
from types import SimpleNamespace
import ipdb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from .CL_Benchmarks import *

######## BASE MODEL FOR FUNCTIONS THAT ARE USED ACROSS DIFFERENT NETWORK ARCHITECTURES #######

class BaseModel(pl.LightningModule):
    def __init__(self, params: SimpleNamespace):
        super().__init__()
        self.automatic_optimization = False 
        self.params = params
        self.save_hyperparameters({"empty":None})
        
        # this is hacky but pytorch lightning doesnt seem to give me any way to do this that is not fitting the model... 
        self.curr_ep = 0

    def init_cl_baselines(self):
        # setting the continual learning baseline regularizer here so it works across models
        if self.params.continual_learning and self.params.cl_baseline is not None:

            if self.params.cl_baseline == "MAS":
                self.cl_baseline_regularization_module = MAS( (self.params, self, self.params.mas_importance) )
            elif self.params.cl_baseline == "EWC-MEMORY":
                self.cl_baseline_regularization_module = EWC_Memory( (self.params, self, self.params.ewc_memory_importance) )
            elif self.params.cl_baseline == "SI":
                self.cl_baseline_regularization_module = SI( (self.params, self, self.params.si_importance) )
            elif self.params.cl_baseline == "L2":
                self.cl_baseline_regularization_module = L2( (self.params, self, self.params.l2_importance) )
            else:
                raise Exception("This CL baseline does not exist.")

    def reload_neuron_activation_counters(self):
        
        if self.params.use_top_k and "GABA_SWITCH" in self.params.k_approach: 
            if not self.params.alex_net_freeze_layer_swap:

                full_cpkt = torch.load(self.params.load_from_checkpoint, map_location=self.device)
                self.sdm_module.top_k.neuron_activation_counters = full_cpkt["neuron_activation_counters"]

                # as it is memory intensive to keep this stored. 
                del full_cpkt

    ############ STATIC FUNCTIONS ############
    def enforce_positive_weights(self):
        # WARNING: this means the interneuron bias terms (and any others I introduce later) cant be negative removing a firing threshold for the interneuron.
        for p in list(self.parameters()):
            # check if p is a bias term or global threshold.
            if len(p.shape) == 1 and not self.params.can_enforce_positive_weights_on_bias_terms:
                continue
                # otherwise wont do anything here to these neurons. 
            p.data.clamp_(0)

    def return_curr_ep(self):
        # used in TopK activation to give current epoch
        if self.params.epoch_relative_to_training_restart:
            return self.curr_ep - self.params.starting_epoch
        return self.curr_ep
            
    def log_wandb(self, dic, commit=False):
        if self.params.logger:
            assert type(dic) == dict, "Need to input a dictionary"
            # Custom function that adds the epoch (and whatever else in the future.) 
            # Also allows for commit to be false to only force it later. 
            dic['epoch'] = self.curr_ep
            wandb.log(dic,commit=commit)
        else: 
            pass
            #print( dic)

    ###### OPTIMIZER #######
    def configure_optimizers(self, verbose=False):
        if self.params.opt == "SGDM":
            optimizer = optim.SGD(list(self.parameters()), lr=self.params.lr, momentum=self.params.sgdm_momentum)

        elif self.params.opt == "SGD":
            optimizer = optim.SGD(list(self.parameters()), lr=self.params.lr) # not defining momentum, defaults to 0. 

        elif self.params.opt == "Adam":
            optimizer = optim.Adam(list(self.parameters()), lr=self.params.lr)

        elif self.params.opt == "RMSProp":
            optimizer = optim.RMSprop(list(self.parameters()), lr=self.params.lr)

        elif self.params.opt == "AdamW":
            optimizer = optim.AdamW(list(self.parameters()), lr=self.params.lr, weight_decay=self.params.adamw_l2_loss_weight)

        else:
            raise NotImplementedError("Need to implement optimizer")
        
        if verbose:
            print("length of net parameters", len(list(self.parameters())))
        return optimizer

    ######### LOSS #########
    def compute_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)

    def extra_loss_terms(self):
        extra_loss = 0
        if self.params.continual_learning and self.params.cl_baseline is not None:
            if self.trainer.datamodule.curr_index>0: 
                # thus it is not the first task. 
                extra_loss += self.cl_baseline_regularization_module.penalty()

        return extra_loss

    ###### TRAIN + VAL #######
    def on_train_start(self, *args):
        # Need to update params in cases where loading in model. 
        if self.params.logger:
            self.params.logger.experiment.config.update(self.params.__dict__, allow_val_change=True) 

        # annoying this base model is called before anything else so need to init this here. 
        self.init_cl_baselines()

        if self.params.load_from_checkpoint:
            self.reload_neuron_activation_counters()

        if self.params.logger:
            self.params.run_id = wandb.run.id

    def on_train_epoch_start(self, *args):
        self.train_loggers = {"train/loss":0}
        self.train_loggers["train/accuracy"]=0

        #Update the model parameters here!
        if self.params.continual_learning and self.params.cl_baseline is not None:
            if self.trainer.datamodule.curr_index>0 and self.trainer.current_epoch % self.params.epochs_per_dataset == 0:
                # it is not the first task and the task is now being incremented!

                self.cl_baseline_regularization_module.learn_task(self.trainer.datamodule.train_datasets[:self.trainer.datamodule.curr_index], self.trainer.datamodule.curr_index)
                
    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = train_batch

        if self.params.investigate_cont_learning: 
            # need to ultimately iterate through all of the training data!!
            # need to save out these values at the end of every epoch? and log the percentages. can then plot which neurons are responding to each value
            logits, model_data_dict = self.forward(x, output_model_data=True)
            self.investigate_cont_learning_obj.update( y, model_data_dict['post_acts'])
         
        elif self.params.model_style.name == "ACTIVE_DENDRITES":
            logits  = self.forward(x, task_info=y)
        else:
            logits = self.forward(x)
            if type(logits) == tuple: 
                logits, model_data_dict = logits
        loss = self.compute_loss(logits, y)

        if self.params.cl_baseline is not None and self.params.cl_baseline == "SI":
            self.cl_baseline_regularization_module.get_unreg_gradients(logits, y)

        loss += self.extra_loss_terms()

        # reverses the mean operation so apply it later at the end. 
        self.train_loggers['train/loss']+=loss.item()*len(x)

        acc = ( y==torch.argmax(logits,dim=1) ).type(torch.float).sum() 
        self.train_loggers['train/accuracy']+=acc
        
        self.manual_backward(loss)

        if self.params.log_gradients and self.curr_ep>=self.params.start_epoch_log_grads:

            log_gradients_obj = LogGradObj(
                torch.clone(self.sdm_module.fc1.weight.detach()),
                model_data_dict['post_acts'].sum(0),
            )

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.params.gradient_clip)

        opt.step()

        if self.params.cl_baseline is not None and self.params.cl_baseline == "SI":
            self.cl_baseline_regularization_module.update_weight_importances()

        if self.params.log_gradients and self.curr_ep>=self.params.start_epoch_log_grads:
            log_gradients_obj.update( self.sdm_module.fc1.weight.detach() )
            
        return loss

    def on_train_epoch_end(self, *args):
        for k,v in self.train_loggers.items():
            self.train_loggers[k] = v/len(self.trainer.datamodule.train_data)
        self.log_wandb(copy.deepcopy(self.train_loggers))

        if self.params.investigate_cont_learning and self.current_epoch % self.params.investigate_cont_learning_log_every_n_epochs == 0:
            # reset the label counter!!! 

            self.investigate_cont_learning_obj.save_and_reset( self.params.test_name, self.current_epoch)

        # should go last. 
        self.curr_ep += 1

    def on_validation_epoch_start(self, *args):
        self.val_loggers = {"val/loss":[0,0]}
        self.val_loggers["val/accuracy"] =[0,0]
        if self.params.continual_learning:
            for i in range(self.trainer.datamodule.curr_index+1):
                self.val_loggers["val/accuracy_split_"+str(i)]=[0,0]
        if self.params.validation_neuron_logger:
            self.post_activation_storage = None
            self.data_labels = []

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y = val_batch
        if self.params.model_style.name == "ACTIVE_DENDRITES":
            logits  = self.forward(x, task_info=y)
        else: 
            out = self.forward(x, output_model_data=self.params.validation_neuron_logger)
            if self.params.validation_neuron_logger:
                logits, model_data_dict = out
            else: 
                logits = out
        loss = self.compute_loss(logits, y)

        loss += self.extra_loss_terms()

        # this reverses the mean operation so can divide overall
        self.val_loggers["val/loss"][0]+=loss.item()*len(x)
        self.val_loggers["val/loss"][1]+=len(x)

        acc = ( y==torch.argmax(logits,dim=1) ).type(torch.float).sum()

        if self.params.continual_learning:
            if dataloader_idx: 
                split_ind = dataloader_idx
            else: 
                split_ind = 0
            self.val_loggers["val/accuracy_split_"+str(split_ind)][0]+=acc
            self.val_loggers["val/accuracy_split_"+str(split_ind)][1]+=len(x)
            
            self.val_loggers["val/accuracy"][0]+=acc*( (self.trainer.datamodule.curr_index+1) / self.params.num_data_splits)
            self.val_loggers["val/accuracy"][1]+=len(x)
            # compensating for the smaller size of datasets. 
        else: 
            self.val_loggers["val/accuracy"][0]+=acc
            self.val_loggers["val/accuracy"][1]+=len(x)

        if self.params.validation_neuron_logger:
            if self.post_activation_storage is None:
                self.post_activation_storage = model_data_dict['post_acts']
            else:
                self.post_activation_storage = torch.cat(
                    [self.post_activation_storage, model_data_dict['post_acts']], dim=0
                )
            self.data_labels += list(y.cpu().numpy())

        return loss

    def time_to_log_images(self):
        return (self.curr_ep+1)%self.params.log_receptive_fields_every_n_epochs==0

    def on_validation_epoch_end(self, *args):
        for k,v in self.val_loggers.items():
            # getting rid of the normalizers.
            self.val_loggers[k] = v[0]/v[1]

        if self.params.logger is None: 
            print(self.val_loggers)
        self.log_wandb(copy.deepcopy(self.val_loggers))
            
        if self.params.validation_neuron_logger:

            ### ACTIVATION INFO using POST_ACTIVATION_STORAGE ###

            # sums w.r.t. batch to get the total activation of each neuron across examples.
            post_activity_per_neuron = self.post_activation_storage.sum(0)
            # NOTE THIS IS ONLY CALCULATED WITH VAL SET EXAMPLES THAT ARE SOMEWHAT OUT OF DISTRIBUTION
            # if a neuron doesn't sufficiently activate across all examples, it is inactive. Compares neuron's accumulated activation to threshold.
            frac_dead_neurons_across_layer_val_dataset = (
                post_activity_per_neuron < self.params.active_threshold
                ).type(torch.float).mean()

            # neurons responding to a given input. how many neurons are active.
            #post_active_per_input = self.post_activation_storage.sum(1)
            frac_neurons_active_per_input_thresh = (
                (self.post_activation_storage > self.params.active_threshold)
                .type(torch.float)
                .mean(1)
            )

            self.log_wandb(
                {
                    "frac_dead_neurons_across_layer_val_dataset":frac_dead_neurons_across_layer_val_dataset,
                    #"total_neuron_activation_per_input": post_active_per_input,
                    "frac_neurons_active_per_input_thresh_val_dataset": frac_neurons_active_per_input_thresh
                }
            )

            if self.time_to_log_images() and self.params.log_receptive_fields: 
                # going to display the 5 most active receptive fields here. 

                if self.params.use_top_k: 
                    model_weights = self.sdm_module.fc1.weight.data
                else: 
                    model_weights = self.fc1.weight.data
                # log the sdm layer (if it exists)
                # if CNN is on then SDM is not only the first layer and so it doesnt make much sense to visualize it. 
                self.log_image( model_weights, "Random Receptive Fields (Addresses)", "Neuron")

                vals, inds = torch.topk(post_activity_per_neuron, 5, dim=0, sorted=False)

                self.log_image( model_weights, "Most Active Receptive Fields (Addresses)", "Most Active Neuron", inds=inds)

            # clear memory
            del self.post_activation_storage
            del self.data_labels
        
    def on_fit_start(self):
        print("Current epoch at fit start is: ", self.curr_ep)
        self.check_enforce_pos_and_l2_norm_weights()
        
        if self.params.investigate_cont_learning:

            self.investigate_cont_learning_obj = InvestigateContLearningObj(  self.params.nneurons[0], self.log_wandb )

        print("Model is:", self)
        
    def on_train_batch_end(self, *args):
        self.check_enforce_pos_and_l2_norm_weights()

    ###### ENFORCE NORMALIZATION ETC #######
    def check_enforce_pos_and_l2_norm_weights(self):
        # this is implemented in the children 
        if self.params.all_positive_weights:
            self.enforce_positive_weights()
        if self.params.norm_addresses or self.params.norm_values:
            self.enforce_l2_norm_weights()
        
        #raise Exception ("Implemented in child classes")
        
    def on_epoch_end(self, *args): 
        # forcing all data to be logged up. 
        # called for train and validation. 
        if self.params.logger:
            wandb.log({"epoch":self.curr_ep},commit=True)

    def on_exception(self, *args):
        self.check_enforce_pos_and_l2_norm_weights()

    def on_save_checkpoint(self, checkpoint):
        self.check_enforce_pos_and_l2_norm_weights()

        checkpoint["hyper_parameters"] = vars(self.params)

        # here I need to store the topK counters throughout the model.
        # find all TopK params in self. Store their neuron activation counters. Need to decide if I should do this for single or multiple.  
        if self.params.use_top_k and "GABA_SWITCH" in self.params.k_approach:
            # TODO, even for single layer case have it be in a list. 
             
            checkpoint["neuron_activation_counters"] = self.sdm_module.top_k.neuron_activation_counters 

            # TODO: have each SDM module save itself? And with the correct index for each layer. Look at how layer norm etc does this saving operaiton under the hood. Or dropout for example. 
            # have this in the on fit start too!!! 

    def log_image(self, weights_to_plot, title, sub_title, inds=None, cnn_weights=False):
        
        if cnn_weights: 
            plot_dim = weights_to_plot.shape[-1]
            num_images = self.params.num_cnn_receptive_field_imgs
        else: 
            plot_dim = self.params.img_dim
            num_images = self.params.num_receptive_field_imgs
        if inds is None:
            inds = range(num_images)
        self.log_wandb(
            {
                title: [
                    wandb.Image(
                        (
                            weights_to_plot[i].reshape(
                                plot_dim, plot_dim
                            )
                            if self.params.nchannels == 1
                            else weights_to_plot[i].reshape(
                                self.params.nchannels,
                                plot_dim,
                                plot_dim,
                            )
                        ),
                        caption=f"{sub_title} #{i}",
                    )
                    for i in inds
                ]
            }
        )
        

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class InvestigateContLearningObj():

    def __init__(self, nneurons, wandb_func, nclasses=10):
        self.log_wandb = wandb_func
        self.nneurons = nneurons
        self.nclasses = nclasses 
        self.split_neuron_activation_label_counter = torch.zeros((nclasses, nneurons, ))
        self.overall_neuron_activation_label_counter = torch.zeros((nclasses, nneurons, ))

    def update(self, y, neuron_activations ):
        for ind in range(len(y)):
            self.split_neuron_activation_label_counter[int(y[ind].cpu().numpy())] += torch.where(neuron_activations[ind]!=0, 1, 0).cpu()

    def save_and_reset(self, test_name, current_epoch):

        self.overall_neuron_activation_label_counter += self.split_neuron_activation_label_counter

        # log these!
        neuron_activation_label_summary_dict = dict(
            split_fraction_activated = (self.split_neuron_activation_label_counter.sum(0)>0).sum() / self.nneurons,
            over_all_splits_fraction_activated = (self.overall_neuron_activation_label_counter.sum(0)>0).sum() / self.nneurons
        )
        self.log_wandb( copy.deepcopy(neuron_activation_label_summary_dict)  )

        self.log_wandb( copy.deepcopy( {"split_neuron_activation_counts":wandb.Histogram(self.split_neuron_activation_label_counter.sum(0)), "overall_neuron_activation_counts":wandb.Histogram(self.overall_neuron_activation_label_counter.sum(0), num_bins=100) } )  )
        

        neuron_activation_label_full_dict = dict(
            split_fraction_activated = self.split_neuron_activation_label_counter,
            over_all_splits_fraction_activated = self.overall_neuron_activation_label_counter
        )

        pickle.dump(neuron_activation_label_full_dict, open(f'../scratch_link/Foundational-SDM/pickles/{test_name}_{current_epoch}_neuron_activation_label_counter.pkl', 'wb') )

        # need to save this out as a pickle too
        self.split_neuron_activation_label_counter = torch.zeros((self.nclasses, self.nneurons, ))


class LogGradObj():

    def __init__(self, pre_update_weights, batch_neuron_acts):
        self.pre_update_weights = pre_update_weights
        self.batch_neuron_acts = batch_neuron_acts
        self.dead_neuron_mask = (batch_neuron_acts==0.0)

    def update(self, post_update_weights):
        weight_deltas = torch.abs(self.pre_update_weights - post_update_weights)
        # computing delta between weights before the step and after the step. 
        if self.dead_neuron_mask.sum()>0: 
            # there is at least one dead neuron:
            wandb.log({
                    'log_grads/dead_neuron_grad_update_l2_norm':torch.norm(weight_deltas[self.dead_neuron_mask],dim=1),
                    'log_grads/mean_dead_neuron_grad_update_l2_norm':torch.norm(weight_deltas[self.dead_neuron_mask],dim=1).mean(),
                },
                commit=True)

        # logging for the alive neurons now. 
        wandb.log({
                'log_grads/alive_neuron_grad_update_sum':weight_deltas[~self.dead_neuron_mask].sum(1), 
                'log_grads/alive_neuron_grad_update_l2_norm':torch.norm(weight_deltas[~self.dead_neuron_mask],dim=1),
                'log_grads/mean_alive_neuron_grad_update_l2_norm':torch.norm(weight_deltas[~self.dead_neuron_mask],dim=1).mean(),
                'log_grads/dead_neurons_in_batch':torch.sum(self.dead_neuron_mask),
            },
            commit=True)

        # now doing 5 random neurons. 
        for nind in range(5):
            neuron_alive = float(self.batch_neuron_acts[nind]>0)
            wandb.log({
                f'neuron_grads_{nind}/Neuron_active':neuron_alive, 
                f'neuron_grads_{nind}/Neuron_grad_update_sum':weight_deltas[nind].sum().item(),
                f'neuron_grads_{nind}/Neuron_grad_update_l2':torch.norm(weight_deltas[nind]).item(),
            },
            commit=True)