#%% guided search
from hyperopt import hp, fmin, tpe, space_eval, rand, Trials, atpe
from case_model_to_tune import case_model_to_tune
from config import space, data_file, num_trials, number_of_features

trials = Trials()
best = fmin(fn=case_model_to_tune, space=space, algo=tpe.suggest, max_evals=len(trials) + 50, trials=trials, catch_eval_exceptions=True)

counter = len(trials)  # the largest possible number for the min loss the remained unchanged
for loss in trials.losses():
    if loss > min(trials.losses()):
        counter -= 1
    else:
        break

print('in this tuning, the best loss has not changed for {}/{} trials and the loss history is'.format(counter, len(trials)), trials.losses())

