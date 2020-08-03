from hyperopt import hp, fmin, tpe, space_eval, rand, Trials
import hyperopt.pyll.stochastic
import itertools
import keras
from sklearn import metrics
import keras.backend as K
import numpy as np
#%% actual config
data_file = 'data.dat'
tune_frequency = 15
sliding_window_size = 30  # this is the amount of data we use for testing
num_trials = None  # number of trials autoML should make to figure out a hyperparameter combination
number_of_features = 4  # number of features that are in data. Important for us to do feature selection.
ratio_of_test_data_in_tuning = 0.04  # during tuning, this percentage of data will be used for validating
number_of_test_data_in_tuning = 7  # during tuning, this percentage of data will be used for validating
num_epochs = 50  # number of epochs for both tuning and testing

target_column_index = 0
out_batch_size = 1
days_of_trace_back_for_features = [0, 0, float('-inf'), float('-inf')]

resumable_file = 'resumable_file'
total_prediction_loss_timestamp_record = 'total_prediction_loss_timestamp.csv'

clipLeft = 0
clipRight = 99999999

data_length = 30

history_length = 10

num_thread_historical = 2
num_thread_random     = 2
num_trials_random     = 2
num_thread_guided     = 2
num_trials_guided     = 2
number_iterations     = 2


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
    hp.choice('feature_indices', indice_list[-1:]),
    hp.choice('in_batch_size', [3,4,5,6,7,8,9,10,11,12,13]),
    hp.choice('steps_ahead', [1]),
    hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    hp.choice('num_cells', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    hp.lognormal('learning_rate', np.log(.01), 3.),
    hp.choice('cnn_filters', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]),
    hp.choice('cnn_kernel_size', [1,3,5,7,9]),
    hp.choice('cnn_pool_size', [1,2,3,4,5,6]),
    hp.choice('cnn_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
]

#%%


