# given a data, find out the most suitable hyperparameters for this data
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, TimeDistributed, Flatten, Dropout, MaxPooling1D, Conv1D
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import pickle
from config import data_file, num_epochs, out_batch_size, days_of_trace_back_for_features, \
    target_column_index, train_loss, subrun_test_length_per_window, calculate_loss_future
from config import saveModel, seed, subrun_sliding_window_number, noSequenceNormalization
import config

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %%
def subRun(hyperparameters):
    with open(data_file, "rb") as f:
        dfSite = pickle.load(f)

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

    # %% feature selection
    padded_feature_indices_with_clinic = [0] + [feature_index + 1 for feature_index in feature_indices]
    dfSite = dfSite.iloc[:, list(padded_feature_indices_with_clinic)]

    days_of_trace_back_for_features_selected = [days_of_trace_back_for_features[i] for i in feature_indices]

    # %% transform to supervised time series
    list_loss_future, list_loss_past, list_predict_future, list_actual_future = [], [], [], []

    for i in reversed(range(subrun_sliding_window_number)):

        # %% split sequences
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

        def normalize_clinic(df):
            # normalize based on the training dataset only
            # split and normalize for each clinic, and do normalization as a whole
            if config.removeNormalizationToMakeDataClearer:
                return df
            else:
                df_normalized = (df.iloc[:, 1:] - df.iloc[:-subrun_test_length_per_window, 1:].mean()) / df.iloc[:-subrun_test_length_per_window, 1:].std()
                return pd.concat([df['clinic'], df_normalized], axis=1)

        ## do normalization
        original_index = dfSite.index
        dfSite_noIndex = dfSite.reset_index(drop=True)
        dfSite_normalized = dfSite_noIndex.groupby('clinic', as_index=False, group_keys=False).apply(normalize_clinic)
        dfSite_normalized.index = original_index

        X_train, y_train, X_test, y_test = [], [], [], []

        for clinic in dfSite_normalized.clinic.unique():
            x_sequences, y_sequences = split_sequences(dfSite_normalized[lambda df: df.clinic == clinic].iloc[:, 1:].values, in_batch_size, steps_ahead, out_batch_size, days_of_trace_back_for_features_selected, target_column_index)

            train_length = len(y_sequences) - subrun_test_length_per_window  # train length in this window

            # based on test_length because the sequence produced by split_sequences might be shorter than train+test length
            if all([len(X_train) == 0, len(y_train) == 0, len(X_test) == 0, len(y_test) == 0]):
                X_train = x_sequences[:-subrun_test_length_per_window]
                y_train = y_sequences[:-subrun_test_length_per_window]
                X_test = x_sequences[-subrun_test_length_per_window:]
                y_test = y_sequences[-subrun_test_length_per_window:]
            else:
                X_train = np.concatenate([X_train, x_sequences[:-subrun_test_length_per_window]])
                y_train = np.concatenate([y_train, y_sequences[:-subrun_test_length_per_window]])
                X_test = np.concatenate([X_test, x_sequences[-subrun_test_length_per_window:]])
                y_test = np.concatenate([y_test, y_sequences[-subrun_test_length_per_window:]])

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

        # %% define model
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], len(feature_indices)))
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
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        model.fit(X_train, y_train, epochs=num_epochs, verbose=0)
        model.save('models/{}'.format(str(hyperparameters)))

        # %%
        y_actual_past = y_train
        y_predict_past = model.predict(x=X_train, verbose=0)

        y_actual_past_1D = y_actual_past.reshape(-1)
        y_predict_past_1D = y_predict_past.reshape(-1)

        loss_past = calculate_loss_future(y_true=y_predict_past_1D, y_pred=y_actual_past_1D)
        list_loss_past.append(loss_past)

        # %% test future n days
        y_actual_future = y_test
        y_predict_future = model.predict(x=X_test, verbose=0)
        y_predict_future = y_predict_future

        y_future_actual_1D = y_actual_future.reshape(-1)
        y_future_predict_1D = y_predict_future.reshape(-1)
        loss_future = calculate_loss_future(y_true=y_future_actual_1D, y_pred=y_future_predict_1D)

        list_loss_future.append(loss_future)

        list_predict_future.append(y_future_predict_1D[-1])
        list_actual_future.append(y_future_actual_1D[-1])

    # %%
    return (sum(list_loss_future) / subrun_sliding_window_number)
# %% example
# hyperparameters = ((0, 3), 9, 1, 0.2, 0.6, 10, 0.0007451801984717818, 160, 2, 1, 0.8)
# print(subRun(hyperparameters))
