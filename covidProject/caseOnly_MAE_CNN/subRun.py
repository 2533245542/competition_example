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
from keras.layers import Dense, TimeDistributed, Flatten, Dropout, MaxPooling1D, Conv1D
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
    cnn_kernel_size = min(in_batch_size, cnn_kernel_size)
    cnn_pool_size = min(in_batch_size - cnn_kernel_size + 1, cnn_pool_size)
    hyperparameters[8] = cnn_kernel_size
    hyperparameters[9] = cnn_pool_size

    # %%
    num_prediction_to_make = number_of_test_data_in_tuning
    sliding_window_size = 1

    # %% feature selection
    data = data[:, list(feature_indices)]
    days_of_trace_back_for_features_selected = [days_of_trace_back_for_features[i] for i in feature_indices]
    data_copy = data.copy()  # the cannon data value

    # %% transform to supervised time series
    list_loss_future = []
    list_loss_past = []
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
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], len(feature_indices)))
        if test_length != 0:
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], len(feature_indices)))
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'), input_shape=(None, in_batch_size, len(feature_indices))))
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
        list_loss_past.append(loss_past)
        if test_length == 0:
            return loss_past

        # %% test future n days
        y_predict_future = model.predict(x=X_test, verbose=0)
        y_predict_future = y_predict_future * y_train_std + y_train_mean
        y_actual_future = y_test * y_train_std + y_train_mean

        # post processing
        y_future_predict_1D = pd.Series(y_predict_future.reshape(-1)).apply(lambda x: x if x > 0 else 0)
        y_future_actual_1D = pd.Series(y_actual_future.reshape(-1))
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
    return (sum(list_loss_future) / len(list_loss_future))
# %% example
# hyperparameters = ((0, 3), 9, 1, 0.2, 0.6, 10, 0.0007451801984717818, 160, 2, 1, 0.8)
# subRun(hyperparameters)


