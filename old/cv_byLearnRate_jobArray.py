import json
import sys
import numpy as np
import datetime

# import user-defined functions

from BaseModel import BaseModel
from GlobalModel import GlobalModel
from IndividualModel import IndividualModel
from FedModel import FedModel
from UserDayData import UserDayData
from collections import defaultdict
from ExperimentUtils import ExperimentUtils
from typing import List, Dict


def run_cv(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    epochs: List[float],
    lrs: List[float],
    parameter_dict: Dict[str, float]
)->Dict[str, float]:

    num_val_samples = (71 * parameter_dict['cv']) // k

    # save metrics in dictionary
    metrics_by_lr = defaultdict(list)

    for epoch in epochs:
        metrics_by_lr[str(epoch)] = defaultdict(list)
        # for cross-validation purposes,
        # use current epoch instead of user-inputted epoch
        parameter_dict['epochs'] = epoch

        for i in range(k):
            val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
            val_fold = train_data.get_subset_for_days(val_days)

            train_days = (
                list(range(i * num_val_samples)) +
                list(range((i + 1) * num_val_samples, num_val_samples * k))
            )
            train_fold = train_data.get_subset_for_days(train_days)

            for lr in lrs:
                # for cross-validation purposes,
                # use current learn_rate instead of user-inputted learn rate
                parameter_dict['learn_rate'] = lr

                model = model_class(
                    parameter_config=parameter_dict,
                )

                results = ExperimentUtils.run_single_experiment(
                    model, train_fold, val_fold)

                # import pdb
                # pdb.set_trace()

                # get the evaluation metric, based on what prediction problem we're doing
                if parameter_dict['output_layer']['loss_type'] == 'regression':
                    metrics_by_lr[str(epoch)][lr].append(results['mse'])
                elif parameter_dict['output_layer']['loss_type'] == 'classification':
                    if len(
                    parameter_dict['output_layer']['classification_thresholds']
                ) == 1:
                        metrics_by_lr[str(epoch)][lr].append(results['AUC'])
                    elif len(
                    parameter_dict['classification_thresholds']
                ) > 1:
                        metrics_by_lr[str(epoch)][lr].append(results['accuracy'])
                    else:
                        raise ValueError('If loss_type is classification, \
                            classification_thresholds in the user-inputted .json \
                            must be a string of at least length 1')
                        
                # write to file
                #ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + 'tmp_cv_lr')
                output_path = parameter_dict['output_path'] + 'tmp_cv_lr'
                with open(output_path + ".json", "w") as f:
                     json.dump(metrics_by_lr, f, indent=4)
        
    for epoch in epochs:
        metrics_by_lr[str(epoch) + '_avg'] = defaultdict(list)
        for lr in lrs:
            metrics_by_lr[str(epoch)+ '_avg'][str(lr) + '_avg'] = np.mean(metrics_by_lr[str(epoch)][lr])

    return metrics_by_lr


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_registry = {
        'individual_model': IndividualModel,
        'global_model': GlobalModel,
        'fed_model': FedModel,
    }
    
    if model_registry.get(parameter_dict.get('model_type')) is None:
        raise KeyError("model_type in .json must be one of: 'individual_model', 'global_model', 'fed_model'")
    else:
        model_class = model_registry[parameter_dict['model_type']]

    k = 3
      #lrs = [0.00001, 0.00005, 0.005, 0.01, 0.03, 0.05] # an hour on jessica's computer
    # lrs = [0.02, 0.03, 0.04, 0.06] # 30 min on jessica's computer
    # lrs = np.linspace(0,1,50, endpoint = False) # 2.55 hours for global model on (0,1,50) on jessica's computer
    # lrs = np.linspace(0,1,25, endpoint = False) # 3.6 hours for individiual model on (0,1,25) on jessica's computer, 2.65 hours for global model + fed model on (0,1,25)
    # lrs = np.linspace(0,0.25,25, endpoint = False)
    #lrs = np.arange(0.05, 0.3,0.01)
    #lrs = [1e-10, 1e-08, 1e-06, 1e-05, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    #lrs = np.arange(0.01, 0.05, 0.01)
    #lrs = np.logspace(-5, -1, base = 10, num = 25)
    #lrs = [1e-10, 1e-08, 1e-06, 1e-05]
    #lrs = np.concatenate([np.arange(0.005,0.01,0.001), np.arange(0.01,0.05,0.01)])
    #import pdb
    #pdb.set_trace()

    #n = len(sys.argv[2]) 
    #a = sys.argv[2][1:n-1] 
    #a = a.split(',') 
      
    #lrs = [float(i) for i in a]
    lrs = [float(sys.argv[3])]

    # tune number of epochs jointly with learning rates
    #epochs = np.arange(10,80,20)
    #epochs = np.concatenate([np.arange(5,25,5), [30]])
    
    #n = len(sys.argv[3]) 
    #a = sys.argv[3][1:n-1] 
    n = len(sys.argv[4]) 
    a = sys.argv[4][1:n-1] 
    a = a.split(', ') 
      
    epochs = [int(i) for i in a] 
    #epochs = [int(sys.argv[4])]
    output_path = sys.argv[5] + sys.argv[6]


    train_data, test_data = ExperimentUtils.simple_train_test_split(parameter_dict)

    metrics_by_lr = run_cv(model_class, train_data, k, epochs, lrs, parameter_dict)


    # output path is now the job array ID
    #ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + '_cv_lr')
    ExperimentUtils.write_to_json(metrics_by_lr, output_path + '_cv_lr')

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))

if __name__ == '__main__':
    main()

