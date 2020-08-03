# given a data, find out the most suitable hyperparameters for this data
import warnings
# warnings.simplefilter(action='ignore', category=Warning)

import os
import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, TimeDistributed, Flatten, Dropout, MaxPooling1D, Conv1D, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import math
import pickle
from config import data_file, ratio_of_test_data_in_tuning, num_epochs, out_batch_size, days_of_trace_back_for_features, \
    target_column_index, train_loss, number_of_test_data_in_tuning, calculate_loss_future

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %%
def subRun(hyperparameters):
    print(hyperparameters)
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    feature_indices = hyperparameters[0]
    in_batch_size = hyperparameters[1]
    steps_ahead = hyperparameters[2]
    dropout = hyperparameters[3]
    recurrent_dropout = hyperparameters[4]
    num_cells = hyperparameters[5]
    learning_rate = hyperparameters[6]
    cnn_filters = hyperparameters[7]
    cnn_kernel_size = hyperparameters[8]
    cnn_pool_size = hyperparameters[9]
    cnn_dropout = hyperparameters[10]
    cnn_pool_size = max(cnn_kernel_size, cnn_pool_size)

    # %%
    num_prediction_to_make = number_of_test_data_in_tuning
    sliding_window_size = 1

    # %% feature selection
    data = data[:, list(feature_indices)]
    days_of_trace_back_for_features_selected = [days_of_trace_back_for_features[i] for i in feature_indices]
    data_copy = data.copy()  # the cannon data value

    # %% transform to supervised time series
    list_loss_future = []  # from day 01/28/2020
    list_predict_future = []
    list_actual_future = []

    for i in reversed(range(sliding_window_size)):
        seed = 2
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        # tf.set_random_seed(seed)
        random.seed(seed)

        data = data_copy[:len(data_copy) - i]  # window slided data
        test_length = num_prediction_to_make  # test length in this window

        # %% split sequences
        def split_sequences(data, in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features,
                            target_column_index):
            '''usage
            split given data to supervised sequenses
            from the sequences then
            need to manually define data collection lag for each feature
            specify the target feature
            usage
            '''

            x_sequences = []
            y_sequences = []

            for i in range(len(data)):
                ## calculate start and end index for features and prediction target
                startIndices = []
                endIndices = []

                ### first do it for features
                for feature_index in range(len(days_of_trace_back_for_features)):
                    if days_of_trace_back_for_features[feature_index] < 0:  # for infinite tracebackable features
                        startIndices.append(i + steps_ahead - 1 + out_batch_size - in_batch_size + 1)
                        endIndices.append(i + steps_ahead - 1 + out_batch_size + 1)
                    else:  # for finite traceback features
                        startIndices.append(i - days_of_trace_back_for_features[feature_index] - in_batch_size + 1)
                        endIndices.append(i - days_of_trace_back_for_features[feature_index] + 1)

                    if endIndices[-1] > len(data) - days_of_trace_back_for_features[
                        feature_index]:  # upper bound for this feature
                        continue

                ### then for the prediction target
                startIndices.append(i + steps_ahead)
                endIndices.append(i + steps_ahead + out_batch_size)
                if endIndices[-1] > len(data) - days_of_trace_back_for_features[target_column_index]:
                    continue

                ## upper and lower bound for data. Skip if any start index < 0  or end index > len(data)
                if sum([indice < 0 for indice in startIndices] + [indice > len(data) for indice in endIndices]) > 0:
                    continue

                ## now we select data
                feature_sequences = []

                for feature_index in range(len(startIndices) - 1):
                    feature_sequences.append(
                        data[startIndices[feature_index]: endIndices[feature_index], feature_index])

                feature_sequences = np.column_stack(feature_sequences)
                x_sequences.append(feature_sequences)
                y_sequences.append(data[startIndices[-1]: endIndices[-1], target_column_index])
            return np.array(x_sequences), np.array(y_sequences)

        x_sequences, y_sequences = split_sequences(data, in_batch_size, steps_ahead, out_batch_size,
                                                   days_of_trace_back_for_features_selected, target_column_index)
        x_index, y_index = split_sequences(
            np.column_stack([list(range(len(data)))] * len(days_of_trace_back_for_features_selected)), in_batch_size,
            steps_ahead, out_batch_size, days_of_trace_back_for_features_selected, target_column_index)

        train_length = len(y_sequences) - test_length  # train length in this window

        X_train, y_train = x_sequences[:train_length], y_sequences[:train_length]  # based on test_length because the sequence produced by split_sequences might be shorter than train+test length
        X_test, y_test = x_sequences[train_length:], y_sequences[train_length:]
        X_train_index, y_train_index = x_index[:train_length], y_index[:train_length]  # useful for plotting
        X_test_index, y_test_index = x_index[train_length:], y_index[train_length:]  # useful for plotting

        # %% regularization
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        y_train_mean = np.mean(y_train, axis=0)
        y_train_std = np.std(y_train, axis=0)

        X_train = (X_train - X_train_mean) / X_train_std
        y_train = (y_train - y_train_mean) / y_train_std

        X_test = (X_test - X_train_mean) / X_train_std
        y_test = (y_test - y_train_mean) / y_train_std

        # %% define model
        # X_train = X_train.reshape((X_train.shape[0], np.sqrt(in_batch_size).astype('int'), np.sqrt(in_batch_size).astype('int'), len(feature_indices)))
        #
        # if test_length != 0:
        #     X_test = X_test.reshape((X_train.shape[0], np.sqrt(in_batch_size).astype('int'), np.sqrt(in_batch_size).astype('int'), len(feature_indices)))
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'), input_shape=X_train.shape[1:]))
        model.add(TimeDistributed(MaxPooling1D(pool_size=cnn_pool_size)))
        model.add(Dropout(cnn_dropout))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(num_cells, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Dense(out_batch_size))
        model.compile(optimizer=Adam(lr=learning_rate), loss=train_loss)

        # %% fit model
        history = model.fit(X_train, y_train, epochs=num_epochs, verbose=0)

        # %% draw
        y_actual_past = y_train * y_train_std + y_train_mean
        y_actual_past_1D = pd.Series(y_actual_past.reshape(-1))

        y_predict_past = model.predict(x=X_train, verbose=0)
        y_predict_past = y_predict_past * y_train_std + y_train_mean
        y_predict_past_1D = pd.Series(y_predict_past.reshape(-1)).apply(lambda x: x if x > 0 else 0)

        loss_past = calculate_loss_future(y_true=y_predict_past_1D, y_pred=y_actual_past_1D)
        list_loss_future.append(loss_past)
        if test_length == 0:
            return loss_past

        # %% test future n days
        y_predict_future = model.predict(x=X_test, verbose=0)
        y_predict_future = y_predict_future * y_train_std + y_train_mean
        y_actual_future = y_test * y_train_std + y_train_mean

        # post processing
        y_future_predict_1D = pd.Series(y_predict_future.reshape(-1)).apply(lambda x: x if x > 0 else 0)
        y_future_actual_1D = pd.Series(y_actual_future.reshape(-1))
        # todo change it to msle
        loss_future = calculate_loss_future(y_true=y_future_actual_1D, y_pred=y_future_predict_1D)

        list_loss_future.append(loss_future)

        list_predict_future.append(y_future_predict_1D.tolist()[-1])
        list_actual_future.append(y_future_actual_1D.tolist()[-1])

        draw_actual_future = y_test * y_train_std + y_train_mean
        draw_predict_future = model.predict(x=X_test, verbose=0) * y_train_std + y_train_mean

        # %% draw
        # fig, axes = plt.subplots(2, 2, dpi=100, figsize=[19.4, 10.8])
        # # fitted curve
        # pd.DataFrame({'predict': y_predict_past[:, -1], 'target': y_actual_past[:, -1]}).plot(title='draw past, predict vs actual, loss_mae ' + str(loss_mae), ax=axes[0,0])
        # # loss vs val_loss curve
        # pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}).plot(ax=axes[0,1])
        # # test
        # pd.DataFrame({'predict': draw_predict_future[:, -1], 'target': draw_actual_future[:, -1]}).plot(title='draw future, loss_mae ' + str(loss_mae), ax=axes[1,0])
        # plt.show()

    # %%
    # print('MAE_n_day_avg_window_size_{}'.format(sliding_window_size), sum(list_loss_future) / sliding_window_size)
    return (sum(list_loss_future) / sliding_window_size)

# %% example
hyperparameters = ((0, 3), 9, 1, 0.2, 0.6, 10, 0.0007451801984717818, 160, 2, 1, 0.8)
subRun(hyperparameters)










#%%
X_train = np.array([[[-1.26572389, -1.08012345], [ 0.02941176, -1.14300114], [-1.03923048, -1.27872403], [ 1.98636658, -1.47196014], [-0.77015405, -1.74371458], [-0.86154979, -2.15900216], [ 0.29672121, -0.9258201 ], [ 0.80622577, -1.08012345], [-0.37947332, -1.14300114]], [[ 0.19689038, -1.08012345], [-1.11764706, -1.14300114], [ 1.96299092, -1.27872403], [-0.75241158, -1.47196014], [-0.77015405, -1.74371458], [ 0.53846862, -0.50800051], [ 0.72531852, -0.9258201 ], [-0.40311289, -1.08012345], [-0.79056942, -1.14300114]], [[-0.90007032, -1.08012345], [ 1.94117647, -1.14300114], [-0.66395281, -1.27872403], [-0.75241158, -1.47196014], [ 0.59511904, -0.23249528], [ 1.00514142, -0.50800051], [-0.5604734 , -0.9258201 ], [-0.80622577, -1.08012345], [ 0.85381497, -1.14300114]], [[ 2.02515822, -1.08012345], [-0.73529412, -1.14300114], [-0.66395281, -1.27872403], [ 0.42135049,  0.        ], [ 1.05021006, -0.23249528], [-0.39487699, -0.50800051], [-0.98907071, -0.9258201 ], [ 0.80622577, -1.08012345], [-2.0238577 , -1.14300114]], [[-0.53441675, -1.08012345], [-0.73529412, -1.14300114], [ 0.46188022,  0.23249528], [ 0.81260451,  0.        ], [-0.31506302, -0.23249528], [-0.86154979, -0.50800051], [ 0.72531852, -0.9258201 ], [-2.01556444, -1.08012345], [ 0.85381497, -1.14300114]], [[-0.53441675, -1.08012345], [ 0.41176471,  0.50800051], [ 0.83715789,  0.23249528], [-0.36115756,  0.        ], [-0.77015405, -0.23249528], [ 1.00514142, -0.50800051], [-2.27486263, -0.9258201 ], [ 0.80622577, -1.08012345], [ 0.85381497,  0.50800051]], [[ 0.56254395,  0.9258201 ], [ 0.79411765,  0.50800051], [-0.28867513,  0.23249528], [-0.75241158,  0.        ], [ 1.05021006, -0.23249528], [-2.2615682 , -0.50800051], [ 0.72531852, -0.9258201 ], [ 0.80622577,  0.9258201 ], [-0.79056942,  0.50800051]], [[ 0.92819752,  0.9258201 ], [-0.35294118,  0.50800051], [-0.66395281,  0.23249528], [ 0.81260451,  0.        ], [-2.13542713, -0.23249528], [ 1.00514142, -0.50800051], [ 0.72531852,  1.08012345], [-0.80622577,  0.9258201 ], [-0.37947332,  0.50800051]], [[-0.16876319,  0.9258201 ], [-0.73529412,  0.50800051], [ 0.83715789,  0.23249528], [-1.92617365,  0.        ], [ 1.05021006, -0.23249528], [ 1.00514142,  1.14300114], [-0.98907071,  1.08012345], [-0.40311289,  0.9258201 ], [ 0.85381497,  0.50800051]], [[-0.53441675,  0.9258201 ], [ 0.79411765,  0.50800051], [-1.78978583,  0.23249528], [ 0.81260451,  0.        ], [ 1.05021006,  1.27872403], [-0.86154979,  1.14300114], [-0.5604734 ,  1.08012345], [ 0.80622577,  0.9258201 ], [ 0.03162278,  0.50800051]], [[ 0.92819752,  0.9258201 ], [-1.88235294,  0.50800051], [ 0.83715789,  0.23249528], [ 0.81260451,  1.47196014], [-0.77015405,  1.27872403], [-0.39487699,  1.14300114], [ 0.72531852,  1.08012345], [ 0.        ,  0.9258201 ], [ 1.67600716,  0.50800051]], [[-1.63137746,  0.9258201 ], [ 0.79411765,  0.50800051], [ 0.83715789,  1.74371458], [-0.75241158,  1.47196014], [-0.31506302,  1.27872403], [ 1.00514142,  1.14300114], [-0.13187609,  1.08012345], [ 1.61245155,  0.9258201 ], [-1.20166551,  0.50800051]], [[ 0.92819752,  0.9258201 ], [ 0.79411765,  2.15900216], [-0.66395281,  1.74371458], [-0.36115756,  1.47196014], [ 1.05021006,  1.27872403], [ 0.07179582,  1.14300114], [ 1.58251314,  1.08012345], [-1.20933866,  0.9258201 ], [ 0.44271887,  2.15900216]]])
y_train = np.array([[-0.64044476], [ 0.87333376], [-1.77577865], [ 0.87333376], [ 0.87333376], [-0.64044476], [-0.26200013], [ 0.87333376], [ 0.1164445 ], [ 1.63022303], [-1.01888939], [ 0.49488913], [-1.39733402]])
X_train.shape  # (13, 9, 2)
y_train.shape  # (13, 1)

X_train = X_train.reshape((13, 1, 9, 2))
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=160, kernel_size=2, activation='relu'), input_shape=(None, 9, 2)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(Dropout(0.8))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.6))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.0007), loss=train_loss)
model.summary()

history = model.fit(X_train, y_train, epochs=num_epochs, verbose=0)
#%%

# this one works, i add an empty dimension to X_train second axis, the input shape is changed to 9
# hyperparameters = ((0, 2, 3), 14, 1, 0.8, 0, 20, 0.11615535071081376, 30, 7, 6, 0.4)
# hyperparameters = ((0, 2, 3), 13, 1, 0.8, 0.1, 30, 0.0002154554090510231, 90, 1, 4, 0.3)
# hyperparameters = ((0, 2, 3), 13, 1, 0.3, 0, 10, 0.025841118651544426, 30, 5, 2, 0.5)
# hyperparameters = ((0, 2, 3), 10, 1, 0.4, 0.6, 10, 0.11411973180282345, 20, 3, 5, 0.3)
# hyperparameters = ((0, 2, 3), 10, 1, 0.4, 0, 60, 0.005741822648467566, 90, 1, 2, 0.8)
# hyperparameters = ((0, 2, 3), 5, 1, 0.6, 0, 90, 0.0066146305489640685, 50, 9, 1, 0.4)
# hyperparameters = ((0, 2, 3), 8, 1, 0.3, 0.5, 60, 0.22019143530401328, 100, 3, 4, 0.4)
# hyperparameters = ((0, 2, 3), 4, 1, 0.7, 0.2, 70, 0.02628300840148588, 30, 5, 2, 0.5)
# hyperparameters = ((0, 2, 3), 8, 1, 0.8, 0.4, 60, 0.0030980154277048693, 60, 3, 3, 0)
# hyperparameters = ((0, 2, 3), 9, 1, 0.7, 0.7, 100, 0.0967598838756082, 150, 9, 1, 0.6)
# hyperparameters = ((0, 2, 3), 12, 1, 0.7, 0.1, 60, 0.11634078449653114, 10, 3, 5, 0.8)
hyperparameters = ((0, 2, 3), 7, 1, 0.1, 0.7, 70, 0.007380736399928478, 100, 3, 6, 0.6)
feature_indices = hyperparameters[0]
in_batch_size = hyperparameters[1]
steps_ahead = hyperparameters[2]
dropout = hyperparameters[3]
recurrent_dropout = hyperparameters[4]
num_cells = hyperparameters[5]
learning_rate = hyperparameters[6]
cnn_filters = hyperparameters[7]
cnn_kernel_size = hyperparameters[8]
cnn_pool_size = hyperparameters[9]
cnn_dropout = hyperparameters[10]
'''
rules:
kernel_size > pool_size
in_batch_szie > kernel_size + 2
'''
cnn_kernel_size = min(in_batch_size, cnn_kernel_size)
cnn_pool_size = min(in_batch_size - cnn_kernel_size + 1, cnn_pool_size)
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'), input_shape=(None, in_batch_size, len(feature_indices))))
model.add(TimeDistributed(MaxPooling1D(pool_size=cnn_pool_size)))
# model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=9, activation='relu'), input_shape=(None, 9, len(feature_indices))))
# model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(Dropout(cnn_dropout))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(num_cells, dropout=dropout, recurrent_dropout=recurrent_dropout))
model.add(Dense(out_batch_size))
model.compile(optimizer=Adam(lr=learning_rate), loss=train_loss)
''':arg
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn1.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn2.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn3.csv

cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn1.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn2.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn3.csv

cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn4.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn5.csv

cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn4.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn5.csv
'''

'''
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm1.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm2.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm3.csv

cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm1.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm2.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm3.csv

cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm4.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm5.csv

cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm4.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm5.csv
'''




''':arg
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn1.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn2.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn3.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn4.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn5.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn6; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn6.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn7; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn7.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn8; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn8.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn9; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn9.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN1; rm resumable_file historicalHyperparameter; python run.py > cnn10; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn10.csv

cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn1.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn2.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn3.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn4.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn5.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn6; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn6.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn7; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn7.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn8; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn8.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn9; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn9.csv
cd ~/tmp/case_only_MAE30/caseOnly_MAE_CNN30; rm resumable_file historicalHyperparameter; python run.py > cnn10; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_cnn10.csv

cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm1.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm2.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm3.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm4.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm5.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm6; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm6.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm7; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm7.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm8; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm8.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm9; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm9.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_validation30; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > clstm0; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm10.csv

cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm1; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm1.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm2; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm2.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm3; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm3.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm4; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm4.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm5; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm5.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm6; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm6.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm7; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm7.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm8; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm8.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > lstm9; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm9.csv
cd ~/tmp/case_only_MAE_fix_feature/caseOnly_MAE_fix_feature_long_validation1; rm resumable_file historicalHyperparameter; python caseOnly_resumable_revolutionalized_fix_length.py > clstm0; mv total_prediction_loss_timestamp.csv total_prediction_loss_timestamp_lstm10.csv



'''

