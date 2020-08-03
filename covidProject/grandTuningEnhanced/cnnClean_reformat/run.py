import config

from multiprocessing import set_start_method

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import tuneWorker
from config import run_sliding_window_number, num_epochs, tune_interval, out_batch_size, days_of_trace_back_for_features, target_column_index, resumable_file, total_prediction_loss_timestamp_record, train_loss, run_time_series_length_for_training_testing, calculate_loss_future, clipLeft, clipRight
from config import transfer_learning_sites, seed, run_test_length_per_window, excludeTargetSiteInTransferLearning, target_site, noSequenceNormalization, testing
from sklearn import metrics
import pickle

tf.get_logger().setLevel('ERROR')
#%%
def run():

    # %% disable cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #%% get data
    covidDf = pd.read_csv('covid_calls_05062020.csv')
    covidDf.columns = ['date', 'clinic', 'call', 'case']

    def convert(s):
        return '/'.join([s[:2], s[2:5], s[5:]])

    covidDf['date'] = pd.to_datetime(covidDf.date.apply(convert).apply(str), infer_datetime_format=True)
    covidDf = covidDf.set_index(['date'])

    def resampleFillNa(df):
        return df.resample('D').asfreq().fillna(method='ffill')

    dfSite = covidDf[lambda df: df.clinic.isin(transfer_learning_sites)].groupby('clinic', group_keys=False, as_index=False).apply(resampleFillNa)

    dfSite['dayofweek'] = dfSite.index.dayofweek
    dfSite['weekofyear'] = dfSite.index.weekofyear
    dfSite['year'] = dfSite.index.year

    dfSite = dfSite[['clinic', 'case', 'call', 'dayofweek', 'weekofyear', 'year']]

    if testing:
        dfSite.case = dfSite.dayofweek
        dfSite.call = dfSite.dayofweek

    dfSite = dfSite.loc[dfSite.index.unique()[clipLeft:clipRight]]

    testDataTimeStamps = dfSite.index.unique()[-run_sliding_window_number:]

    #%% transform to supervised time series
    list_predict_future, list_actual_future, list_loss_past, list_loss_future, list_hyperparameters = [], [], [], [], []

    hyperparameters, feature_indices, in_batch_size, steps_ahead, dropout, recurrent_dropout, num_cells, learning_rate, cnn_filters, cnn_kernel_size, cnn_pool_size, cnn_dropout = [None] * 12  # need to make variabels contain something to make it picklable

    iteration_list = list(reversed(range(run_sliding_window_number)))

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
            cnn_filters = pickle.load(f)
            cnn_kernel_size = pickle.load(f)
            cnn_pool_size = pickle.load(f)
            cnn_dropout = pickle.load(f)

            list_predict_future = pickle.load(f)
            list_actual_future = pickle.load(f)
            list_loss_past = pickle.load(f)
            list_loss_future = pickle.load(f)
            testDataTimeStamps = pickle.load(f)
            list_hyperparameters = pickle.load(f)
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
            pickle.dump(cnn_filters, f)
            pickle.dump(cnn_kernel_size, f)
            pickle.dump(cnn_pool_size, f)
            pickle.dump(cnn_dropout, f)

            pickle.dump(list_predict_future, f)
            pickle.dump(list_actual_future, f)
            pickle.dump(list_loss_past, f)
            pickle.dump(list_loss_future, f)
            pickle.dump(testDataTimeStamps, f)
            pickle.dump(list_hyperparameters, f)

        test_length = run_test_length_per_window  # test length in this window

        #%% auto tune
        dfSite_window = dfSite.loc[dfSite.index.unique()[ len(dfSite.index.unique()) - i - run_time_series_length_for_training_testing: len( dfSite.index.unique()) - i]]
        if i == (run_sliding_window_number - 1) or (run_sliding_window_number - 1 - i) % tune_interval == 0: # first window or meets a sliding interval

            tuneType = True if i == (run_sliding_window_number - 1) else False

            if excludeTargetSiteInTransferLearning:
                dfSite_window = dfSite_window[lambda df: ~(df.clinic == target_site)]

            hyperparameters = tuneWorker.doTune(dfSite_window, tuneType)  # pass a subset of dfSite data

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

        ## feature selection
        padded_feature_indices_without_clinic = [feature_index + 1 for feature_index in feature_indices]
        data = dfSite_window[lambda df: df.clinic == target_site].iloc[:, list(padded_feature_indices_without_clinic)].values  # now we need to explicitly select 663 site
        days_of_trace_back_for_features_selected = [days_of_trace_back_for_features[i] for i in feature_indices]

        #%% split sequence
        def split_sequences(data, in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features,
                            target_column_index):

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

        ## normalization
        site_mean = np.mean(data[:-test_length], axis=0)
        site_std = np.std(data[:-test_length], axis=0)

        if config.removeNormalizationToMakeDataClearer:
            site_mean = site_mean * 0
            site_std = site_std * 0 + 1

        data = (data - site_mean) / site_std

        x_sequences, y_sequences = split_sequences(data, in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features_selected, target_column_index)

        X_train, y_train = x_sequences[:-run_test_length_per_window], y_sequences[:-run_test_length_per_window]
        X_test, y_test = x_sequences[-run_test_length_per_window:], y_sequences[-run_test_length_per_window:]

        # %% regularization
        if not noSequenceNormalization:
            X_train_mean = np.mean(X_train, axis=0)
            X_train_std = np.std(X_train, axis=0)
            y_train_mean = np.mean(y_train, axis=0)
            y_train_std = np.std(y_train, axis=0)

            X_train = (X_train - X_train_mean) / X_train_std
            y_train = (y_train - y_train_mean) / y_train_std

            X_test = (X_test - X_train_mean) / X_train_std
            y_test = (y_test - y_train_mean) / y_train_std

        # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], len(feature_indices)))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], len(feature_indices)))


        # load the model here
        model = load_model('models/{}'.format(str(hyperparameters)))
        model.layers[0].trainable = False   # CNN
        model.layers[1].trainable = True
        model.layers[2].trainable = True
        model.layers[3].trainable = True
        model.layers[4].trainable = False   # LSTM
        model.layers[5].trainable = True    # Dense
        model.compile(optimizer=Adam(lr=learning_rate), loss=train_loss)

        #%% fit model
        os.environ['PYTHONHASHSEED'] = str(seed)  # seed before training for reproducibility from the tuned result
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, verbose=0)

        #%% draw
        y_predict_past = model.predict(x=X_train, verbose=0)
        if noSequenceNormalization:
            y_actual_past = y_train
            y_predict_past = y_predict_past
        else:
            y_actual_past = y_train * y_train_std + y_train_mean
            y_predict_past = y_predict_past * y_train_std + y_train_mean

        y_actual_past = y_actual_past * site_std[0] + site_mean[0]
        y_predict_past = y_predict_past * site_std[0] + site_mean[0]

        y_actual_past = np.maximum(y_actual_past.reshape(-1), 0)
        y_predict_past = np.maximum(y_predict_past.reshape(-1), 0)

        loss_past = metrics.mean_squared_log_error(y_true=y_actual_past, y_pred=y_predict_past)
        list_loss_past.append(loss_past)

        # %% test future n days
        y_predict_future = model.predict(x=X_test, verbose=0)
        if noSequenceNormalization:
            y_predict_future = y_predict_future
            y_actual_future = y_test
        else:
            y_predict_future = y_predict_future * y_train_std + y_train_mean
            y_actual_future = y_test * y_train_std + y_train_mean

        y_predict_future = y_predict_future * site_std[0] + site_mean[0]
        y_actual_future = y_actual_future * site_std[0] + site_mean[0]

        # post processing
        y_predict_future = np.maximum(y_predict_future.reshape(-1), 0)
        y_actual_future = np.maximum(y_actual_future.reshape(-1), 0)

        loss_future = calculate_loss_future(y_true=y_actual_future, y_pred=y_predict_future)
        list_loss_future.append(loss_future)
        list_predict_future.append(y_predict_future[-1])
        list_actual_future.append(y_actual_future[-1])
        list_hyperparameters.append(str(hyperparameters))

        #%% plot
        fig, axes = plt.subplots(2, 2, dpi=100, figsize=[19.4, 10.8])

        # fitted curve
        pd.DataFrame({'predict': y_predict_past[-300:, 0], 'target': y_actual_past[-300:, 0]}).plot(title='draw past, predict vs actual, loss_past ' + str(loss_past), ax=axes[0,0])
        # loss vs val_loss curve
        pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}).plot(ax=axes[0,1])
        # prediction curve
        pd.DataFrame({'predict': y_predict_future[:, 0], 'target': y_actual_future[:, 0]}).plot(title='draw future, predict vs actual, loss_past ' + str(loss_future), ax=axes[1,0])
        pd.DataFrame({'record hyperparameters': [0]}).plot(title=str(hyperparameters), ax=axes[1, 1])

        plt.savefig('plots/{}_diagnosis_plot_{}.png'.format(target_site, run_sliding_window_number - i), dpi=200)
        print('completed {}/{} sliding window, current MAE {}, MAE so far {}'.format(run_sliding_window_number - i, run_sliding_window_number, calculate_loss_future(y_true=list_actual_future[-1:], y_pred=list_predict_future[-1:]), calculate_loss_future(y_true=list_actual_future, y_pred=list_predict_future)), 'Hyperparamters using:', hyperparameters)

    pd.DataFrame({'testDataTimeStamps': testDataTimeStamps, 'list_predict_future': list_predict_future,
                  'list_actual_future': list_actual_future, 'list_loss_future': list_loss_future,
                  'list_loss_past': list_loss_past, 'list_hyperparameters': list_hyperparameters}).to_csv(total_prediction_loss_timestamp_record)

    print('MAE_n_day_avg_window_size_{}'.format(run_sliding_window_number), sum(list_loss_future) / run_sliding_window_number)

    pd.DataFrame({'actual number of phone calls related to COVID-19 symtoms': list_actual_future, 'predicted number of phone calls related to COVID-related symtoms': list_predict_future}, index=testDataTimeStamps).plot(title='site {}, MAE={}'.format(target_site, calculate_loss_future(y_true=list_actual_future, y_pred=list_predict_future)))
    plt.savefig('plots/{}.png'.format(target_site), dpi=200)


if __name__ == '__main__':
    set_start_method("spawn")
    run()
