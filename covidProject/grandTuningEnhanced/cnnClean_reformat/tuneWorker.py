# usage: need to change the number_of_features to proper values, model_tuned should not have the line of running it
import warnings
# warnings.simplefilter(action='ignore', category=Warning)

from multiprocessing import set_start_method
# set_start_method('forkserver', force=True)
import os
import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
import random
from hyperopt import hp, fmin, tpe, space_eval, rand, Trials, atpe, trials_from_docs
from hyperopt.fmin import generate_trials_to_calculate
from subRun import subRun
import pickle
import itertools
import hyperopt.pyll.stochastic
from config import space, data_file, num_trials, number_of_features, num_thread_random, num_trials_random, num_thread_guided, num_trials_guided, regularTuneIterations, reevaluate_history_length, num_thread_historical
import numpy as np
import multiprocessing
import random
from config import lessTuningIteration, longTuneIterations, tuneWorkerCatchException
import config
#%% tune and run
historical_hyperparameter_history = 'historicalHyperparameter'
historicalRecordDir = 'historicalRecord'
randomRecordDir = 'randomRecord'
guidedRecordDir = 'guidedRecord'

# random.seed(1)


# global thread_random_search

def thread_historical_search(id, trials):
    fmin(fn=subRun, space=space, algo=rand.suggest, max_evals=1,
         trials=trials, catch_eval_exceptions=tuneWorkerCatchException)
    with open('{}/{}'.format(historicalRecordDir, id), 'wb') as f:
        pickle.dump(trials, f)

def thread_random_search(id):
    trials = Trials()
    fmin(fn=subRun, space=space, algo=rand.suggest, max_evals=len(trials) + num_trials_random,
         trials=trials, catch_eval_exceptions=tuneWorkerCatchException)
    with open('{}/{}'.format(randomRecordDir, id), 'wb') as f:
        pickle.dump(trials, f)

# global thread_guided_search
def thread_guided_search(id, trials):
    fmin(fn=subRun, space=space, algo=tpe.suggest, max_evals=len(trials) + num_trials_guided,
         trials=trials, catch_eval_exceptions=tuneWorkerCatchException)
    with open('{}/{}'.format(guidedRecordDir, id), 'wb') as f:
        pickle.dump(trials.trials[-num_trials_guided:], f)

def doTune(data, longTune=True):
    if longTune:
        tune_interations = longTuneIterations
    else:
        tune_interations = regularTuneIterations

    #%% pass data
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    #%% auto tuning
    if not os.path.exists(historicalRecordDir):
        os.makedirs(historicalRecordDir)

    if not os.path.exists(randomRecordDir):
        os.makedirs(randomRecordDir)

    if not os.path.exists(guidedRecordDir):
        os.makedirs(guidedRecordDir)

    # id pool
    max_id = 0  # id counter
    id_pool_random = []
    id_pool_guided = []
    # load previous hyperparamters and generate baseline historical trials record
    id_pool_historical = []
    historical_trials = Trials()
    if os.path.exists(historical_hyperparameter_history):
        with open(historical_hyperparameter_history, 'rb') as f:
            historical_hyperparameters = pickle.load(f)  # historical_hyperparameters = [{'a': [1], 'c1': [], 'c2': [-0.13475270138215323]}]
            # split trials

            adapted_num_thread_historical = min(len(historical_hyperparameters), num_thread_historical)
            smallTrialsList = []
            sub_length = len(historical_hyperparameters) // adapted_num_thread_historical

            for i in range(adapted_num_thread_historical):
                if i == adapted_num_thread_historical - 1:
                    smallTrialsList.append(generate_trials_to_calculate(historical_hyperparameters))
                else:
                    smallTrialsList.append(generate_trials_to_calculate(historical_hyperparameters[:sub_length]))
                    historical_hyperparameters = historical_hyperparameters[sub_length:]

            pool = multiprocessing.Pool(processes=adapted_num_thread_historical)
            for smallTrials in smallTrialsList:
                id = max_id
                max_id += 1
                id_pool_historical.append(id)
                pool.apply_async(thread_historical_search, args=(id, smallTrials))

            pool.close()
            pool.join()

            # merge trials, assign to historical_trials
            for id in id_pool_historical:
                with open('{}/{}'.format(historicalRecordDir, id), 'rb') as f:
                    historical_trials = trials_from_docs(list(historical_trials) + list(pickle.load(f)))

    for _ in range(tune_interations):
        # random phase
        pool = multiprocessing.Pool(processes=num_thread_random)

        for _ in range(num_thread_random):
            id = max_id
            max_id += 1
            id_pool_random.append(id)
            pool.apply_async(thread_random_search, args=(id,))

        pool.close()
        pool.join()

        # loading random and guided pool
        merged_trials = Trials()
        for id in id_pool_random:
            with open('{}/{}'.format(randomRecordDir, id), 'rb') as f:
                merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

        for id in id_pool_guided:
            with open('{}/{}'.format(guidedRecordDir, id), 'rb') as f:
                merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

        # merge merged trials with historical trials
        merged_trials = trials_from_docs(list(merged_trials) + list(historical_trials))
        # guided phase
        pool = multiprocessing.Pool(processes=num_thread_guided)

        for _ in range(num_thread_guided):
            id = max_id
            max_id += 1
            id_pool_guided.append(id)
            pool.apply_async(thread_guided_search, args=(id, merged_trials))

        pool.close()
        pool.join()

    # aggregate
    merged_trials = Trials()
    for id in id_pool_random:
        with open('{}/{}'.format(randomRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    for id in id_pool_guided:
        with open('{}/{}'.format(guidedRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    # merge with historical trials
    merged_trials = trials_from_docs(list(merged_trials) + list(historical_trials))

    # extract hyperparemters
    best = fmin(fn=subRun, space=space, algo=tpe.suggest, max_evals=0, trials=merged_trials,
                catch_eval_exceptions=tuneWorkerCatchException, verbose=False)

    hyperparameters = space_eval(space, best)

    # save hyperparameters
    # if historical_trials contains merged_trials.best_trial, updated_historical_trials = historical_trials
    def twoTrialsAreSame(trialA, trialB):
        same = True
        for key in trialA['misc']['vals']:
            if trialA['misc']['vals'][key][0] != trialB['misc']['vals'][key][0]:
                same = False

        return same

    bestTrialAlreadyIncluded = False
    for trial in historical_trials.trials:
        if twoTrialsAreSame(trial, merged_trials.best_trial):
            bestTrialAlreadyIncluded = True
    if bestTrialAlreadyIncluded:
        updated_historical_trials = historical_trials
    else:
        updated_historical_trials = trials_from_docs(list([merged_trials.best_trial]) + list(historical_trials))

    with open(historical_hyperparameter_history, 'wb') as f:
        # get a list of trials, sort trials by loss, select the top history_length ones, get the parameter-value dictionary
        historical_hyperparameters = [h['misc']['vals'] for h in sorted(updated_historical_trials.trials, key=lambda k: k['result']['loss'])[: reevaluate_history_length]]
        for historical_hyperparameter in historical_hyperparameters:  # unbox the singleton list, to make it ready for producing Trials object
            for key in historical_hyperparameter:
                historical_hyperparameter[key] = historical_hyperparameter[key][0]

        pickle.dump(historical_hyperparameters, f)

    return hyperparameters

#%%
# hyperopt.pyll.stochastic.sample(space)
