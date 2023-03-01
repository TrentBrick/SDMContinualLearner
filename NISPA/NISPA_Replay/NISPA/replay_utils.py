import copy
import numpy as np
import torch
from continuum import rehearsal
from continuum.tasks import TaskSet
from torch.utils.data import TensorDataset


class Memory:
    def __init__(self, train_scenario, memo_size, nb_total_classes):
        self.memory = rehearsal.RehearsalMemory(
                memory_size=memo_size,
                herding_method="random",
                fixed_memory=0,
                nb_total_classes = nb_total_classes
            )
        self.train_scenario = copy.deepcopy(train_scenario)

    def push_penultimate(self, model, train_loader, task_id):
        model.train()
        all_activations = []
        targets = []
        task = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                _, activations = model.forward_activations(data, None)
                all_activations.append(activations[-2])
                targets.append(target)
                task.append([task_id]* len(target))
        mem_x = np.concatenate([np.array(acti.cpu()) for acti in all_activations])
        mem_y = np.concatenate([np.array(l.cpu()) for l in targets])
        mem_t = np.concatenate([np.array(t) for t in task])
        self.memory.add(mem_x, mem_y, mem_t, z = False)

    def push_samples(self, task_id):
        #Get samples function applies transformation to samples
        #Use only normalization otherwise we will store augmented samples
        self.memory.add(*self.train_scenario[task_id].get_samples(range(len(self.train_scenario[task_id]))), z = False)


    def get_random_samples(self, n):
        l = len(self.memory)
        index = np.random.choice(list(range(l)), size = n, replace = False)
        mem_x, mem_y, mem_t = self.memory.get()
        return mem_x[index], mem_y[index], mem_t[index]

    #Use this to get pytorch dataset for memory samples
    def create_dataset(self):
        mem_x, mem_y, _ = self.memory.get()
        return  TensorDataset(torch.Tensor(mem_x),torch.Tensor(mem_y))



    
    
    