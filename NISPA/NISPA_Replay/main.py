from dataset import get_dataset
from NISPA import learner
from args import get_args
import sys
import os

if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('Data'):
        os.makedirs('Data')

    #Get arguments
    args =  get_args()
    
    
    #Create continuum scenario
    scenario_train, scenario_test, input_dim, output_dim, transform_train, transform_test = get_dataset(args.dataset, increment = args.class_per_task, args= args)
    
    accs_after_finetune = []
    freeze_percs = []
    optimal_ps = []
    task_ps = []
    num_frozen_units = []
    
    learner = learner.Learner(args, input_dim, output_dim, scenario_train, scenario_test, 
                              deterministic = True,
                               train_tranform = transform_train, test_transform =  transform_test)
    

    
    for task_id, taskset in enumerate(scenario_train):
       optimal_p,  percentange_of_frozen_conns, acc_after_fine_tuning, _, list_of_ps, num_important_units =  learner.task_train(task_id, taskset, None)

       task_ps =  task_ps + list_of_ps
       num_frozen_units.append(num_important_units)
       accs_after_finetune.append(acc_after_fine_tuning)
       freeze_percs.append(percentange_of_frozen_conns)
       optimal_ps.append(optimal_p)
       
       
    #Write results   
    import xlsxwriter
    row = 0
    column = 0
    workbook = xlsxwriter.Workbook('./results/{}.xlsx'.format(args.experiment_name))
    worksheet = workbook.add_worksheet()
    setting =  ' '.join(sys.argv[1:])
    worksheet.write(row, column, "Results: {}".format(setting))
    #Write selected ps
    column += 1
    row = 0
    worksheet.write(row, column, "Final P")
    for i in range(len(optimal_ps)):
        row += 1
        worksheet.write(row, column, optimal_ps[i])
        
    #Write After accs
    column += 1
    row = 0
    worksheet.write(row, column, "Test Accuracy")
    for i in range(len(accs_after_finetune)):
        row += 1
        worksheet.write(row, column, accs_after_finetune[i])
        
    workbook.close()