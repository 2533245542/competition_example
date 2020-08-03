from hyperopt import hp, fmin, tpe, space_eval, rand, Trials
import hyperopt.pyll.stochastic
import itertools
import keras
from sklearn import metrics
import keras.backend as K
import numpy as np
#%% actual config
tune_interval = 15
run_sliding_window_number = 15  # this is the amount of data we use for testing
subrun_test_length_per_window = 1  # during tuning, this percentage of data will be used for validating. This cannot be 0
run_time_series_length_for_training_testing = 30

reevaluate_history_length = 1
num_thread_historical     = 1
num_thread_random         = 1
num_trials_random         = 1
num_thread_guided         = 1
num_trials_guided         = 1
regularTuneIterations     = 1
longTuneIterations        = 1

########################################################################################################################################################################################################
# new vars
excludeTargetSiteInTransferLearning = False

target_site = '663'
doTransferLearning = True
saveModel = True
loadModel = True
noSequenceNormalization = True
lessTuningIteration = True

## test related
testing = False
removeNormalizationToMakeDataClearer = False
showWarnings = False
tuneWorkerCatchException = False
transfer_learning_sites = [
    '436', '438', '463','504', '506', '508', '516', '526', '528', '531', '537', '539', '541', '544', '546', '548',
         '549', '550', '553', '554', '556', '561', '590', '608', '621', '626', '632', '635', '642', '644', '646', '658', '659',
         '662', '663', '668', '673', '674', '678', '679', '688', '693', '695', '756']

########################################################################################################################################################################################################
data_file = 'data.dat'
num_trials = None  # number of trials autoML should make to figure out a hyperparameter combination
number_of_features = 4  # number of features that are in data. Important for us to do feature selection.
ratio_of_test_data_in_tuning = 0.04  # during tuning, this percentage of data will be used for validating
num_epochs = 50  # number of epochs for both tuning and testing

target_column_index = 0
out_batch_size = 1
days_of_trace_back_for_features = [0, 0, float('-inf'), float('-inf')]

resumable_file = 'resumable_file'
total_prediction_loss_timestamp_record = 'total_prediction_loss_timestamp.csv'
clipLeft = 0
clipRight = 10000  # must be positive numbers

run_test_length_per_window = 1
subrun_sliding_window_number = 1

seed = 2
# def train_loss(y_true, y_pred):
#    # y_pred = K.tf.cast(y_pred > 0, y_pred.dtype) * y_pred  # set negative y_pred to 0
#    y_pred = K.cast(y_pred > 0, y_pred.dtype) * y_pred  # set negative y_pred to 0
#    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
#    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
#    return K.mean(K.square(first_log - second_log), axis=-1)

# def MAPE(y_true, y_pred):
#     y_true = np.clip(a=y_true, a_min=1e-7, a_max=None)
#     y_pred = np.clip(a=y_pred, a_min=1e-7, a_max=None)
#
#     output_errors = np.average(np.abs((y_true - y_pred) / y_true)) * 100
#     return output_errors

# train_loss = train_loss  # used in model.compile
# calculate_loss_future = metrics.mean_squared_log_error   # used in calculating loss to return

train_loss = keras.losses.mean_absolute_error  # used in model.compile
calculate_loss_future = metrics.mean_absolute_error   # used in calculating loss to return


#%% derived config
indice_list = []
for i in range(number_of_features+1):
    indice_list += list(itertools.combinations(list(range(number_of_features)), i))

indice_list = [indices for indices in indice_list if 0 in indices]

space = [
    hp.choice('feature_indices', [(0,2,3)]),
    hp.uniformint('in_batch_size', 3,7),
    hp.choice('steps_ahead', [1]),
    hp.uniform('dropout', 0, 0.8),
    hp.uniform('recurrent_dropout', 0,0.8),
    hp.uniformint('num_cells', 10, 160),
    hp.lognormal('learning_rate', np.log(.01), 3.),
    hp.uniformint('cnn_filters', 10, 160),
    hp.uniformint('cnn_kernel_size', 1,9),
    hp.uniformint('cnn_pool_size', 1,6),
    hp.uniform('cnn_dropout', 0,0.8)
]

#%%


