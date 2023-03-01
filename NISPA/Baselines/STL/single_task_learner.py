from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import random
import numpy as np
from model import random_prune, ConvNet, MLP
from config import DEVICE
from training import naive_training, test
from continuum.tasks import split_train_val

def taskset2GPUdataset(task_set):
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

class Single_Task_Learner():
    def __init__(self, args, input_dim, output_dim, scenario_train, scenario_test, deterministic = True):
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
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args
        self.scenario_train = scenario_train
        self.scenario_test = scenario_test
        self.test_accs = []
        self.tasks = []
        
    def train_all(self):
        for task_id, taskset in enumerate(self.scenario_train):
            current_classes = list(taskset.get_classes())
            #Create data loaders
            train_taskset, val_taskset = split_train_val(taskset, val_split=0.1)
            train_loader = DataLoader(taskset2GPUdataset(train_taskset),  batch_size= self.args.batch_size,  shuffle=True)
            val_loader = DataLoader(taskset2GPUdataset(val_taskset),  batch_size= self.args.batch_size,  shuffle=True)
        
    
            test_set = self.scenario_test[task_id]  
            test_loader = DataLoader(taskset2GPUdataset(test_set), batch_size = self.args.batch_size*2, shuffle=False)
        
            #Get training objects
            loss = nn.CrossEntropyLoss()
            
            if self.args.model == 'conv':
               self.model = ConvNet(self.input_dim, self.output_dim).to(DEVICE)
            else:
                self.model = MLP(self.input_dim, self.output_dim).to(DEVICE)
            self.pruned_model = random_prune(self.model, self.args.prune_perc)
            print(self.model)
            
            #Reset optimizer for every task
            optimizer = self.optim_obj(self.pruned_model.parameters(),
                                  lr= self.args.learning_rate,
                                  weight_decay= self.args.weight_decay)
            
            

            
            
            self.pruned_model, _, _, _ = naive_training(self.pruned_model, loss, optimizer, train_loader,
                            val_loader, self.args, 
                            current_classes if self.args.mask_outputs == 1  else None,
                            multihead = bool(self.args.multihead))
            
            _, accuracy_final, _ = test(self.pruned_model, loss, 
                                 test_loader, current_classes = current_classes)
            
            self.tasks.append(current_classes)
            self.test_accs.append(accuracy_final)
            
        with open("./logs/" + self.args.experiment_name + ".txt", "w") as f:
            for task, acc in zip(self.tasks, self.test_accs):
                f.write("\n" + str(task) + ': ' + str(acc))
            f.write("\n Average: " + str(sum(self.test_accs) / len(self.test_accs)))
                
            
        return self.test_accs