#%% multiprocessing search
import multiprocessing
import os
import random
from hyperopt import hp, fmin, tpe, space_eval, rand, Trials, atpe, trials_from_docs
from case_model_to_tune import case_model_to_tune
from config import space, data_file, num_trials, number_of_features
import pickle


#%% set up
num_thread_random = 3
num_trials_random = 2
num_thread_guided = 3
num_trials_guided = 2
number_iterations = 5
randomRecordDir = 'randomRecord'
guidedRecordDir = 'guidedRecord'

if not os.path.exists(randomRecordDir):
    os.makedirs(randomRecordDir)

if not os.path.exists(guidedRecordDir):
    os.makedirs(guidedRecordDir)

# id pool
id_pool_random = []
id_pool_guided = []
for _ in range(number_iterations):
    # random phase
    ## generate id, do small random * num random, save with id
    def small_random_search(id):
        trials = Trials()
        fmin(fn=case_model_to_tune, space=space, algo=rand.suggest, max_evals=len(trials) + num_trials_random, trials=trials, catch_eval_exceptions=True)
        with open('{}/{}'.format(randomRecordDir, id), 'wb') as f:
            pickle.dump(trials, f)

    pool = multiprocessing.Pool(processes=num_thread_random)

    for _ in range(num_thread_random):
        id = random.randint(1, 1111111111)
        id_pool_random.append(id)
        pool.apply_async(small_random_search, args=(id,))

    pool.close()
    pool.join()
    print('random parallel runs finish')

    # loading random and guided pool
    merged_trials = Trials()
    for id in id_pool_random:
        with open('{}/{}'.format(randomRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    for id in id_pool_guided:
        with open('{}/{}'.format(guidedRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    id_pool_random = []
    id_pool_guided = []

    # guided phase
    def small_guided_search(id, trials):
        fmin(fn=case_model_to_tune, space=space, algo=tpe.suggest, max_evals=len(trials) + num_trials_guided, trials=trials, catch_eval_exceptions=True)
        with open('{}/{}'.format(guidedRecordDir, id), 'wb') as f:
            pickle.dump(trials, f)

    pool = multiprocessing.Pool(processes=num_thread_guided)

    for _ in range(num_thread_guided):
        id = random.randint(1, 1111111111)
        id_pool_guided.append(id)
        pool.apply_async(small_guided_search, args=(id, merged_trials))

    pool.close()
    pool.join()
    print('guided parallel runs finish')

    # aggregate
    merged_trials = Trials()
    for id in id_pool_random:
        with open('{}/{}'.format(randomRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    for id in id_pool_guided:
        with open('{}/{}'.format(guidedRecordDir, id), 'rb') as f:
            merged_trials = trials_from_docs(list(merged_trials) + list(pickle.load(f)))

    ## best hyperparemter and its loss
    print(merged_trials.best_trial['misc']['vals'])
    print(merged_trials.best_trial['result']['loss'])

'''
random pool
guided pool
for iteations:
    random search phase, add to random pool
    load random pool and guided pool
    guided sesarch phase, add to guided pool
    aggregate random pool and guided pool and print current best
'''

