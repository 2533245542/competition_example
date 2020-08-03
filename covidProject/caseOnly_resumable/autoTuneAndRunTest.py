# usage: need to change the number_of_features to proper values, model_tuned should not have the line of running it
# import warnings
# warnings.simplefilter(action='ignore', category=Warning)
#
# import os
# import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

from hyperopt import hp, fmin, tpe, space_eval, rand, Trials, atpe
from case_model_to_tune import case_model_to_tune
import pickle
import itertools
import hyperopt.pyll.stochastic
from config import space, data_file, num_trials, number_of_features

#%% tune and run
def autoTuneAndRun(data):
    trials = Trials()

    # pass data
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    # case_model_to_tune(hyperopt.pyll.stochastic.sample(space))
    # automate the given data
    best = fmin(fn=case_model_to_tune, space=space, algo=tpe.suggest, max_evals=len(trials) + num_trials, trials=trials, catch_eval_exceptions=True)
    hyperparameters = space_eval(space, best)
    # print(hyperparameters)

    counter = len(trials)  # the largest possible number for the min loss the remained unchanged
    for loss in trials.losses():
        if loss > min(trials.losses()):
            counter -= 1
        else:
            break

    print('in this tuning, the best loss has not changed for {}/{} trials and the loss history is'.format(counter, len(trials)), trials.losses())
    return hyperparameters

#%% example
# autoTuneAndRun(data)

#%% when we know what hyperparameters to use
# def autoTuneAndRun(data):
#     return ((0, 2, 3, 4), 8, 16, 0.4, 0.7, 40, 0.0048454985283532944)


# #%% random+guided search
# from hyperopt import hp, fmin, tpe, space_eval, rand, Trials, atpe
# from case_model_to_tune import case_model_to_tune
# from config import space, data_file, num_trials, number_of_features
#
# trials = Trials()
# for _ in [1, 1]:
#     best = fmin(fn=case_model_to_tune, space=space, algo=rand.suggest, max_evals=len(trials) + 25, trials=trials, catch_eval_exceptions=True)
#     best = fmin(fn=case_model_to_tune, space=space, algo=tpe.suggest, max_evals=len(trials) + 25, trials=trials, catch_eval_exceptions=True)
#
# counter = len(trials)  # the largest possible number for the min loss the remained unchanged
# for loss in trials.losses():
#     if loss > min(trials.losses()):
#         counter -= 1
#     else:
#         break
#
# print('in this tuning, the best loss has not changed for {}/{} trials and the loss history is'.format(counter, len(trials)), trials.losses())
#
#
#
