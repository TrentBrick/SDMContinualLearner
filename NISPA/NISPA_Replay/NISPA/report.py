from torch.utils.data import DataLoader
from .training import test
import torch
import numpy as np
from .utils import list_of_weights, taskset2GPUdataset
import pickle
from torch.utils.data.dataloader import default_collate
from .config import DEVICE



def get_output_groups(scenario_train):
    return [list(scenario.get_classes()) for scenario in scenario_train]

        
        
def get_output_activations(model, scenario_test, task_id, args):
    #Task activation for corresponding unit
    activations_all = []
    
    #Compute activation across all tasks so far
    activation_across_all_tasks = []
        
    for backward_task_dataset in scenario_test:
        test_loader = DataLoader(backward_task_dataset,batch_size= args.batch_size*2, shuffle=False, collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        activations = 0
        with torch.no_grad():
            model.eval()
            for idx, (data, target, _) in enumerate(test_loader):
                output = model(data, None)
                activation_across_all_tasks.append(output)
                activations = activations + np.sum(np.array(output.cpu()), axis = 0)
        activations_all.append((backward_task_dataset.get_classes(), activations / len(backward_task_dataset)))
    activations_all.append((scenario_test.classes, torch.mean(torch.vstack(activation_across_all_tasks), dim = 0).cpu().detach().numpy()))
    return activations_all

#Write a summary of the state after each task
def task_summary_log(model, loss,  current_classes, args, task_id, scenario_test, stable_units_set):
    weights = list_of_weights(model)
    units = [w.shape[0] for w, _ in weights] 
    if stable_units_set is None:
        frozen_units = [0]
    else:
       frozen_units = [len(s) for s in stable_units_set[1:]]
    
    #Backward Tasks Accuracy
    backward_accs = []
    backward_tasks = []
    for backward_task_id in range(task_id + 1):
        backward_task_dataset = scenario_test[backward_task_id]
        backward_tasks.append(backward_task_dataset.get_classes())
        test_loader = DataLoader(backward_task_dataset,
                                 batch_size= args.batch_size*2,
                                 shuffle=False, collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))
        _, acc1_back, _ = test(model, loss, test_loader,
                               current_classes =  None)
        backward_accs.append(acc1_back)
            
    with open("./logs/" + args.experiment_name + "_TASK_ID_{}".format(task_id + 1) + '.txt', "w") as fout:
        fout.write(str(args))
        cum_acc = sum(backward_accs) / len(backward_accs)
        fout.write("\n" + "Top 1 Test Accuracy (Cumulative: ):$"+ str(cum_acc) + "$")
        fout.write("\n" + "Top 1 Test Accuracy (Current Task: {}):*".format(current_classes) + str(backward_accs[task_id]) + "*")
    
        
        count = 1
        for task, backward_acc in zip(backward_tasks[:-1], backward_accs[:-1]):
            if backward_acc is None:
                continue
            fout.write("\n" + "TASK ID {}  Top 1 Test Accuracy (Backward Task {}: {}):&".format(task_id, count, task) + str(backward_acc) + "&")
            count += 1
        if args.experiment_note != "":
            fout.write("\n" + args.experiment_note)
        fout.write('\n #Units: ' +  str(units))
        fout.write('\n #Stable Units:@' + str(frozen_units) +"@")  
        
    return backward_accs, backward_tasks, cum_acc

        