# multivariate cnn lstm
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import math

seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix - 1 + n_steps_out:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# path = '../editData/663GCdataMultivariateLSTM.csv'
path = '../editData/663GEdataMultivariateLSTM.csv'
# path = '../editData/663A4dataMultivariateLSTM.csv'
# path = '../editData/663GDdataMultivariateLSTM.csv'
# path = '../editData/663GAdataMultivariateLSTM.csv'
# path = '../editData/663dataMultivariateLSTM.csv'
# path = '../editData/663GBdataMultivariateLSTM.csv'

df = pd.read_csv(path)
df = df[['n_walkins', 'dayofweek', 'day', 'month', 'year']]
df = df[(df['dayofweek'] != 5) & (df['dayofweek'] != 6)]
df = df[['n_walkins', 'year']]
data = df.values

# choose a number of time steps
n_steps_in, n_steps_out = 7, 1  # n_steps_out means the number of future steps. 1 means predicting one future
# 7
train_ratio = 1

test_n_day_length = 21
train_length = math.floor((len(data) - test_n_day_length) * train_ratio)
test_length = 0

train_mean = np.mean(data[:train_length], axis=0)
train_std = np.std(data[:train_length], axis=0)

data = (data - train_mean) / train_std

X_train, y_train = split_sequences(data[:train_length], n_steps_in, n_steps_out)
X_test, y_test = split_sequences(data[train_length:train_length + test_length], n_steps_in, n_steps_out)
X_test_n_day, y_test_n_day = split_sequences(data[train_length + test_length:], n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = X_train.shape[2]
# n_seq = 3
# n_steps = 3
# X_train = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))

# define model
model = Sequential()
# model.add(
#     TimeDistributed(Conv1D(filters=80, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
model.add(LSTM(20, activation='relu',  input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.001), loss='mae')

# fit model
# X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))

# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=0)
history = model.fit(X_train, y_train, epochs=20, verbose=0)
# yhat = model.predict(x=X_test, verbose=0)
#
# y_test = y_test * train_std[0] + train_mean[0]
# yhat = yhat * train_std[0] + train_mean[0]
#
# pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}).plot()
# plt.show()
#
# # post processing
# yhat = pd.Series(yhat.reshape(-1)).apply(math.floor).apply(lambda x: x if x > 0 else 0)
# y_test = pd.Series(y_test.reshape(-1))
# loss_mae = np.abs(yhat - y_test).mean()
# print('MAE')
# print(loss_mae)
# # pd.DataFrame({'predict': yhat, 'target': y_test}).plot(title='loss_mae ' + str(loss_mae))
# # plt.show()
# #
# # np.abs(pd.Series([df['n_walkins'][:math.floor(len(df) * train_ratio)].mean()] * y_test.shape[0]) - y_test).mean()
#
# %% test future n days
yhat_n_day = model.predict(x=X_test_n_day, verbose=0)
y_test_n_day = y_test_n_day * train_std[0] + train_mean[0]
yhat_n_day = yhat_n_day * train_std[0] + train_mean[0]

# post processing
yhat_n_day = pd.Series(yhat_n_day.reshape(-1)).apply(math.floor).apply(lambda x: x if x > 0 else 0)
y_test_n_day = pd.Series(y_test_n_day.reshape(-1))
loss_mae_n_day = np.abs(yhat_n_day - y_test_n_day).mean()
print('MAE_n_day')
print(loss_mae_n_day)

