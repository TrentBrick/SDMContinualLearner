from torch.utils.data import DataLoader
from .training import test
from .utils import list_of_weights, taskset2GPUdataset

#Write a summary of the state after each task
def task_summary_log(model, loss, acc_after_drop, current_classes, args, task_id, scenario_test, stable_units_set):
    weights = list_of_weights(model)
    units = [w.shape[0] for w, _ in weights] 
    if stable_units_set is None:
        frozen_units = [0]
    else:
       frozen_units = [len(s) for s in stable_units_set[1:]]
    
    task_test_current = scenario_test[task_id] 
    test_loader = DataLoader(taskset2GPUdataset(task_test_current),
                             batch_size= args.batch_size*2,
                             shuffle=False)
    
    #Backward Tasks Accuracy
    backward_accs = []
    backward_tasks = []
    if task_id == 0:
        backward_accs.append(None)
    else:
        for backward_task_id in range(task_id):
            backward_task_dataset = scenario_test[backward_task_id]
            backward_tasks.append(backward_task_dataset.get_classes())
            test_loader = DataLoader(taskset2GPUdataset(backward_task_dataset),
                                     batch_size= args.batch_size*2,
                                     shuffle=False)
            _, acc1_back, _ = test(model, loss, test_loader,
                                   current_classes = backward_task_dataset.get_classes() if args.multihead else None)
            backward_accs.append(acc1_back)
            
    with open("./logs/" + args.experiment_name + "_TASK_ID_{}".format(task_id + 1) + '.txt', "w") as fout:
        fout.write(str(args))
        
        if task_id != 0:
            fout.write("\n" + "Top 1 Test Accuracy (Cumulative: ): "+ str(sum(backward_accs + [acc_after_drop]) / len(backward_accs + [acc_after_drop])))
        
        fout.write("\n" + "Top 1 Test Accuracy (Current Task After FineTuning: {}): ".format(current_classes) + str(acc_after_drop))
    
        
        count = 1
        for task, backward_acc in zip(backward_tasks, backward_accs):
            if backward_acc is None:
                continue
            fout.write("\n" + "TASK ID {}  Top 1 Test Accuracy (Backward Task {}: {}): ".format(task_id, count, task) + str(backward_acc))
            count += 1
        if args.experiment_note != "":
            fout.write("\n" + args.experiment_note)
        fout.write('\n #Units: ' +  str(units))
        fout.write('\n #Stable Units' + str(frozen_units))  

        