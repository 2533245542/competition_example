# usage: remove resumble_file first
import warnings
# warnings.simplefilter(action='ignore', category=Warning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import math
import tmp
from config import sliding_window_size, num_epochs, tune_frequency, out_batch_size, days_of_trace_back_for_features, target_column_index, resumable_file, total_prediction_loss_timestamp_record, loss, data_length
from sklearn import metrics
import pickle
#%%
def case_model_auto_tuned():
    # %% disable cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #%%
    site = '663'
    num_prediction_to_make = 1

    #%% get data
    covidDf = pd.read_csv('covid_calls_05062020.csv')
    covidDf.columns = ['date', 'clinic', 'call', 'case']

    def convert(s):
        return '/'.join([s[:2], s[2:5], s[5:]])

    covidDf['date'] = pd.to_datetime(covidDf.date.apply(convert).apply(str), infer_datetime_format=True)
    covidDf = covidDf.set_index(['date'])

    dfSite = covidDf[lambda df: df.clinic == site].resample('D').asfreq().fillna(0)
    dfSite['dayofweek'] = dfSite.index.dayofweek
    dfSite['weekofyear'] = dfSite.index.weekofyear
    dfSite['year'] = dfSite.index.year
    dfSite = dfSite[['case', 'call', 'dayofweek', 'weekofyear', 'year']]

    dfSite = dfSite[-data_length:]

    dfSite_backup = dfSite.copy()

    testDataTimeStamps = dfSite_backup.index[-sliding_window_size:]
    #%% transform to supervised time series
    list_predict_future = []
    list_actual_future = []
    list_loss_past = []  # from day 01/28/2020
    list_loss_future = []  # from day 01/28/2020

    hyperparameters, feature_indices, in_batch_size, steps_ahead, dropout, recurrent_dropout, num_cells, learning_rate = 1,1,1,1,1,1,1,1  # need to make variabels contain something to make it picklable

    iteration_list = list(reversed(range(sliding_window_size)))
    if os.path.exists(resumable_file):
        with open(resumable_file, "rb") as f:
            i = pickle.load(f)

            hyperparameters = pickle.load(f)
            feature_indices = pickle.load(f)
            in_batch_size = pickle.load(f)
            steps_ahead = pickle.load(f)
            dropout = pickle.load(f)
            recurrent_dropout = pickle.load(f)
            num_cells = pickle.load(f)
            learning_rate = pickle.load(f)

            list_predict_future = pickle.load(f)
            list_actual_future = pickle.load(f)
            list_loss_past = pickle.load(f)
            list_loss_future = pickle.load(f)
            testDataTimeStamps = pickle.load(f)
        iteration_list = iteration_list[iteration_list.index(i):]

    for i in iteration_list:

        with open(resumable_file, "wb") as f:
            pickle.dump(i, f)

            pickle.dump(hyperparameters, f)
            pickle.dump(feature_indices, f)
            pickle.dump(in_batch_size, f)
            pickle.dump(steps_ahead, f)
            pickle.dump(dropout, f)
            pickle.dump(recurrent_dropout, f)
            pickle.dump(num_cells, f)
            pickle.dump(learning_rate, f)

            pickle.dump(list_predict_future, f)
            pickle.dump(list_actual_future, f)
            pickle.dump(list_loss_past, f)
            pickle.dump(list_loss_future, f)
            pickle.dump(testDataTimeStamps, f)

        seed = 2
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)

        dfSite_tmp = dfSite.copy()
        dfSite_tmp = dfSite_tmp.iloc[:len(dfSite_tmp) - i, :]
        data = dfSite_tmp.values

        test_length = num_prediction_to_make  # test length in this window
        train_length = len(data) - test_length  # train length in this window

        #%% auto tune
        """in here, use 'data[:train_length + test_length]' to tune the model
        return and reset hyperparameters
        """
        if i == (sliding_window_size - 1) or (sliding_window_size - 1 - i) % tune_frequency == 0:
            from importlib import reload
            # del autoTuneAndRun
            # import autoTuneAndRun
            # reload(autoTuneAndRun)

            hyperparameters = tmp.autoTuneAndRun(data[:train_length])
            feature_indices = hyperparameters[0]
            in_batch_size = hyperparameters[1]
            steps_ahead = hyperparameters[2]
            dropout = hyperparameters[3]
            recurrent_dropout = hyperparameters[4]
            num_cells = hyperparameters[5]
            learning_rate = hyperparameters[6]

        ## feature selection
        data = data[:, list(feature_indices)]
        days_of_trace_back_for_features_selected = [days_of_trace_back_for_features[i] for i in feature_indices]

        #%% split sequence
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

        x_sequences, y_sequences = split_sequences(data, in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features_selected, target_column_index)
        x_index, y_index = split_sequences(np.column_stack([list(range(len(data)))] * len(days_of_trace_back_for_features_selected)), in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features_selected, target_column_index)

        X_train, y_train = x_sequences[:-test_length], y_sequences[:-test_length]  # based on test_length because the sequence produced by split_sequences might be shorter than train+test length
        X_test, y_test = x_sequences[-test_length:], y_sequences[-test_length:]
        X_train_index, y_train_index = x_index[:-test_length], y_index[:-test_length]  # useful for plotting
        X_test_index, y_test_index = x_index[-test_length:], y_index[-test_length:]  # useful for plotting

        # %% regularization
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        y_train_mean = np.mean(y_train, axis=0)
        y_train_std = np.std(y_train, axis=0)

        X_train = (X_train - X_train_mean) / X_train_std
        y_train = (y_train - y_train_mean) / y_train_std

        X_test = (X_test - X_train_mean) / X_train_std
        y_test = (y_test - y_train_mean) / y_train_std

        # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

        # define model
        model = Sequential()
        model.add(LSTM(num_cells, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(in_batch_size, len(feature_indices))))
        model.add(Dense(out_batch_size))
        model.compile(optimizer=Adam(lr=learning_rate), loss=loss)

        #%% fit model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, verbose=0)

        #%% draw
        y_actual_past = y_train * y_train_std + y_train_mean
        y_actual_past_1D = pd.Series(y_actual_past.reshape(-1))

        y_predict_past = model.predict(x=X_train, verbose=0)
        y_predict_past = y_predict_past * y_train_std + y_train_mean
        y_predict_past_1D = pd.Series(y_predict_past.reshape(-1)).apply(math.floor).apply(lambda x: x if x > 0 else 0)

        # todo change it to msle
        loss_past = metrics.mean_squared_log_error(y_true=y_actual_past_1D, y_pred=y_predict_past_1D)

        list_loss_past.append(loss_past)
        # %% test future n days
        y_predict_future = model.predict(x=X_test, verbose=0)
        y_predict_future = y_predict_future * y_train_std + y_train_mean
        y_actual_future = y_test * y_train_std + y_train_mean

        # post processing
        y_future_predict_1D = pd.Series(y_predict_future.reshape(-1)).apply(math.floor).apply(lambda x: x if x > 0 else 0)
        y_future_actual_1D = pd.Series(y_actual_future.reshape(-1))
        # todo change it to msle
        loss_future = metrics.mean_squared_log_error(y_true=y_future_actual_1D, y_pred=y_future_predict_1D)


        list_loss_future.append(loss_future)
        list_predict_future.append(y_future_predict_1D.tolist()[-1])
        list_actual_future.append(y_future_actual_1D.tolist()[-1])

        #%% plot
        plt.clf()
        plt.cla()
        plt.close()
        fig, axes = plt.subplots(2, 2, dpi=100, figsize=[19.4, 10.8])

        # fitted curve
        pd.DataFrame({'predict': y_predict_past[-300:, 0], 'target': y_actual_past[-300:, 0]}).plot(title='draw past, predict vs actual, loss_past ' + str(loss_past), ax=axes[0,0])
        # loss vs val_loss curve
        pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}).plot(ax=axes[0,1])
        # prediction curve
        pd.DataFrame({'predict': y_predict_future[:, 0], 'target': y_actual_future[:, 0]}).plot(title='draw future, predict vs actual, loss_past ' + str(loss_future), ax=axes[1,0])
        pd.DataFrame({'record hyperparameters': [0]}).plot(title=str(hyperparameters), ax=axes[1, 1])

        # plt.show()
        plt.savefig('plots/{}_diagnosis_plot_{}.png'.format(site, sliding_window_size - i), dpi=400)
        print('completed {}/{} sliding window,average MAE so far {}'.format(sliding_window_size - i, sliding_window_size, sum(list_loss_future)/len(list_loss_future)), 'hyperparamters using', hyperparameters)

    pd.DataFrame({'testDataTimeStamps': testDataTimeStamps, 'list_predict_future': list_predict_future, 'list_actual_future': list_actual_future, 'list_loss_future': list_loss_future, 'list_loss_past': list_loss_past}).to_csv(total_prediction_loss_timestamp_record)

    print('MAE_n_day_avg_window_size_{}'.format(sliding_window_size), sum(list_loss_future) / sliding_window_size)

    plt.clf()
    plt.cla()
    plt.close()
    pd.DataFrame({'actual number of phone calls related to COVID-19 symtoms': list_actual_future, 'predicted number of phone calls related to COVID-related symtoms': list_predict_future}, index=testDataTimeStamps).plot(title='site {}, MAE={}'.format(site, sum(list_loss_future) / len(list_loss_future)))
    # plt.show()
    plt.savefig('plots/{}.png'.format(site), dpi=400)

case_model_auto_tuned()
