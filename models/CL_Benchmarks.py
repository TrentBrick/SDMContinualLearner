from copy import deepcopy
import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader

# copied and modified from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/regularization.py


class Benchmark_Base_Model(object):
    def __init__(self, agent_config):
        self.model_params, self.model = agent_config[0], agent_config[1]
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.baseline_reg_loss_coef = agent_config[2]
        self.regularization_terms = {}
        self.task_count = 0
        #self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
                                # False: Each task has its own importance matrix and model parameters
        self.num_batches_per_dataset = self.model_params.cl_baseline_batches_per_dataset
        self.batch_size = self.model_params.cl_baseline_batch_size

    def learn_task(self, list_of_datasets: list, curr_data_index):

        self.model.eval()

        print(curr_data_index, self.task_count )

        assert curr_data_index == self.task_count+1, "These should be equal!"

        # 1.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        just_finished_dataset = list_of_datasets[self.task_count]
        train_loader = DataLoader(
            just_finished_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        # 2.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)
        
        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}

    def penalty(self):
        if len(self.regularization_terms)>0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    #print("IMPORTANCE", importance[n].mean())
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss

        #print("REG LOSS", reg_loss)

        return self.baseline_reg_loss_coef * reg_loss

def softmax_cross_entropy(output, y, beta):
    # this would be if it was unsupervised. 
    #label = output.max(1)[1].view(-1)
    if beta == 1.0: 
        out_log_sms = F.log_softmax(output, dim=1)
    else: 
        out_log_sms = torch.log(torch.exp(output*beta)/torch.exp(output*beta).sum(1,keepdim=True))
    return F.nll_loss(out_log_sms, y, reduction='sum')


class L2(Benchmark_Base_Model):
    """
    BaseModel that is used throughout
    """
    def __init__(self, agent_config):
        super(L2, self).__init__(agent_config)
        self.online_reg = True

    def calculate_importance(self, _):
        # Use an identity importance so it is an L2 regularization.
        # this is if it is not used as a base model. 
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

class EWC_Memory(Benchmark_Base_Model):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.online_reg = False
        self.empFI = False
        self.beta=torch.Tensor([self.model_params.ewc_memory_beta]).to(self.model_params.device)

    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        #self.log('Computing EWC')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        self.model.eval()

        # Accumulate the square of gradients
        for i, (input, target) in enumerate(dataloader):
            if i>self.num_batches_per_dataset:
                break
            self.model.zero_grad()

            #import ipdb 
            #ipdb.set_trace()

            input = input.to(self.model_params.device)
            target = target.to(self.model_params.device)

            pred = self.model.forward(input)
            ind = pred.max(1)[1].flatten()  # Choose the one with max

            # Use groundtruth label (default is without this)
            if self.empFI:  
                ind = target

            loss = softmax_cross_entropy(pred, ind, self.beta) 
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    #print("PARAM IMPORTANCE!!!:", (self.params[n].grad ** 2).mean())
                    #import ipdb 
                    #ipdb.set_trace() 
                    p += ((self.params[n].grad ** 2) / (self.num_batches_per_dataset*self.batch_size) )

        self.model.train()
        return importance

class MAS(Benchmark_Base_Model):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, agent_config):
        super(MAS, self).__init__(agent_config)
        self.online_reg = True

    def calculate_importance(self, dataloader):
        #self.log('Computing MAS')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        self.model.eval()

        # Accumulate the gradients of L2 loss on the outputs
        for i, (input, target) in enumerate(dataloader):
            if i>self.num_batches_per_dataset:
                break
            self.model.zero_grad()

            input = input.to(self.model_params.device)
            target = target.to(self.model_params.device)

            pred = self.model.forward(input)
            pred.pow_(2)
            loss = pred.mean()

            
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += (self.params[n].grad.abs() / len(dataloader))

        self.model.train()
        return importance

class SI(Benchmark_Base_Model):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """

    def __init__(self, agent_config):
        super(SI, self).__init__(agent_config)
        self.online_reg = True  # Original SI works in an online updating fashion
        self.damping_factor = 0.1
        self.beta=torch.Tensor([self.model_params.si_beta]).to(self.model_params.device) 
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def get_unreg_gradients(self, logits, y):

        loss = softmax_cross_entropy(logits, y, self.beta)
        
        # 1.Save current parameters
        self.old_params = {}
        for n, p in self.params.items():
            self.old_params[n] = p.clone().detach()

        # 2. Collect the gradients without any regularization term
        '''out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()'''
        # only on the first loss here before extra terms added
        loss.backward(retain_graph=True)

        self.unreg_gradients = {}
        for n, p in self.params.items():
            if p.grad is not None:
                self.unreg_gradients[n] = p.grad.clone().detach()

        # this is the actual update that happens by default. needs to be in the compute penalty

        # 3. Normal update with regularization
        '''loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()'''

    def update_weight_importances(self):
        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - self.old_params[n]
            if n in self.unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= self.unreg_gradients[n] * delta  # w[n] is >=0

    def calculate_importance(self, _):

        #import ipdb
        #ipdb.set_trace()
        #self.log('Computing SI')
        # dont need the dataloader here. 
        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance