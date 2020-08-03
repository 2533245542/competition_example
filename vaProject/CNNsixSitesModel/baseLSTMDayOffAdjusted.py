import sys
import random
import os
import tensorflow as tf
import pandas as pd
from math import floor
import numpy as np  # linear algebra
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt  # this is used for the plot the graph
from keras.optimizers import Adam

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout

# %% hyperparameter
train_ratio = 0.35  # 0.35   loss = 1.63 if train_ratio = 0.7 loss=1.776 if train_ratio = 0.35
# test_ratio = 0.65 # implied

dropout = 0.2
cells = 50
epochs = 80
batch_size = 80
print_convergence = 0
loss = 'mean_squared_error'
optimizer = 'adam'
n_times = 60

path = '../editData/648dataMultivariateLSTM.csv'
# path = '../editData/691dataMultivariateLSTM.csv'
# path = '../editData/674dataMultivariateLSTM.csv'
# path = '../editData/580dataMultivariateLSTM.csv'
# path = '../editData/740dataMultivariateLSTM.csv'

# %% seeding
seed = 200
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

#%% handle arguments
# sys.argv is at least of length 1
# one argument, file name
if len(sys.argv) == 2:
    path = sys.argv[1]

# %% read data
all_df = pd.read_csv(path)
df = pd.read_csv(path)
df = df[['n_walkins']]
data = df.values


# %% transform data


def extract_lag(set_acf, n):
    '''
    I don't understand why the authors uses such a confusing way to extract the index of the best n values of set_acf
    I make a series from set_acf, sort it and return the index of the top n value

    Glue code to extract the most highly correlated ts n lags from the set.
    Returns a list of lags.
    '''
    temp_index = list()
    for i in range(0, len(set_acf)):
        temp_index.append(i)

    temp_index = np.array(temp_index)
    temp_index = temp_index.reshape(len(temp_index), 1)
    set_acf = set_acf.reshape(len(set_acf), 1)

    x = np.concatenate((temp_index, set_acf), axis=1)
    x = x[1:, :]  # not using the first row because the rcf is always 1, not meaningful
    x = pd.DataFrame(x, columns=('index', 'acf'))
    x = x.sort_values(by=['acf'], ascending=False)

    lag_list = x.iloc[0:n, 0]

    lag_list = lag_list.tolist()
    return (lag_list)


walk_in_series = data[:, 0]
set_acf = acf(walk_in_series)
lag_list = extract_lag(set_acf, 7)


def series_to_supervised(data, lag_list, dropnan=True):
    """
    Prepare data of different time stamps
    Return a dataframe where each column is a time stamp, although the columns are unlikely to order chronologically
    :param data:
    :param lag_list:
    :param dropnan:
    :return:
    """
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # runs through lags in lag list and returns shifted columns
    for i in lag_list:
        cols.append(dff.shift(int(i)))  # shift down the data by i to make a lagged version of the data
        names.append('t-%d' % (int(i)))

    # put them all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names  # column names are the time stamp

    agg = agg.join(dff, how='left')

    # drop rows with NaN values
    if dropnan:  # this is very dangerous, as if one of the cols is mostly or all na, all the na rows will be removed, which might account for most of the data
        agg.dropna(inplace=True)
    return agg


def prep_data(set, n_train):
    """
    :param set:
    :param n_train:
    :return:
    """
    values = set.values
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(values[:n_train, :])
    scaled = scaler.transform(values)

    reframed = series_to_supervised(scaled, lag_list, dropnan=True)

    reframed['n_walkins'] = reframed[0]
    reframed.drop([0], axis=1,
                  inplace=True)  # y might be dropped as a duplicate of the timestamp0? Anyways the effect of these two lines are to just let the datafram contain only time stamp columns, nothing else
    values = reframed.values

    train, test = values[:n_train, :], values[n_train:, :]
    test_index = reframed.index[n_train:]  # for referencial purpose. Later we use test_index to match between train/test and df
    return (scaler, train, test, test_index)


scaler, train, test, test_index = prep_data(df, floor(len(df) * train_ratio))


def split_3d(train, test):
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return (train_X, train_y, test_X, test_y)


# split and reshape (lstms expect 3d inputs)
train_X, train_y, test_X, test_y = split_3d(train, test)


# %%

# wrapper for lstm
def fit_lstm(train_X, train_y, test_X, test_y, dropout, cells, epochs, batch_size, print_convergence, loss, optimizer):
    model = Sequential()
    model.add(LSTM(cells, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout))  # adding recurrent dropout actually
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    # stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_X, test_y), verbose=print_convergence, shuffle=False)
    return (model, history)


# fit model
model, history = fit_lstm(train_X=train_X, train_y=train_y,
                          test_X=test_X, test_y=test_y,
                          dropout=dropout, cells=cells,
                          epochs=epochs, batch_size=batch_size,
                          print_convergence=print_convergence,
                          loss=loss, optimizer=optimizer)


# model fit stat against test data
def model_summary(model, test_X, test_y):
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 7))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate mae
    mae = mean_absolute_error(inv_y, inv_yhat)
    return (round(mae, 3))


model_mae = model_summary(model=model, test_X=test_X, test_y=test_y)


def make_pred(model, test_X, test_y):
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 7))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    return inv_y, inv_yhat


# plot predict vs actual
actual, pred, = make_pred(model=model, test_X=test_X, test_y=test_y)

# plt.figure()
# days = range(1, len(actual) + 1)
# plt.plot(days, actual, 'b-', label='Target')
# plt.plot(days, pred, 'g-', label='Prediction')
# plt.title('Target and prediction')
# plt.legend()
# plt.show()


print('mae(before postprocessing)', model_mae)  # 1.769


# %%
def customMAE(predict, target):
    return abs(predict - target).mean()


# %%
postProcessDf = pd.DataFrame({'walk_in_predict': pred, 'walk_in_actual': actual}, index=test_index)

# %% create and transform df
fullInfoDf = pd.merge(postProcessDf, all_df.iloc[:, 1:], left_index=True, right_index=True)

# change negative prediction to be zero
fullInfoDf['walk_in_predict'] = fullInfoDf['walk_in_predict'].apply(lambda x: 0 if x < 0 else x)

# change round all numbers
fullInfoDf['walk_in_predict'] = fullInfoDf['walk_in_predict'].apply(round)

# change weekend prediction to be zero
fullInfoDf.loc[(fullInfoDf['dayofweek'] == 5) | (fullInfoDf['dayofweek'] == 6), ['walk_in_predict']] = 0

# only consider rows that are not inherently 0, so the MAE latter will be more reflective of the reality
fullInfoDf = fullInfoDf[(fullInfoDf['dayofweek'] != 5) & (fullInfoDf['dayofweek'] != 6)]

# %% calculate MAE
print('mae(after postprocessing)', customMAE(fullInfoDf['walk_in_predict'], fullInfoDf['walk_in_actual']))  # 1.769
print('Using path', path)

"""
mae(before postprocessing) 1.767
mae(after postprocessing) 1.576
"""
# %% plot to see
pred_date = pd.to_datetime(
    all_df[['year', 'month', 'day']].apply(lambda x: '-'.join([str(i) for i in x]), axis=1)).rename('date')
plotDf = pd.merge(fullInfoDf, pred_date, left_index=True, right_index=True)
plotDf[['walk_in_predict', 'walk_in_actual', 'date']].plot(x='date', figsize=[6.4, 4.8])
# plt.show()

# plot descent history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Convergence, MAE: {}'.format(customMAE(fullInfoDf['walk_in_predict'], fullInfoDf['walk_in_actual'])))
plt.ylabel('Loss: MSE')
plt.xlabel('Epoch(s)', size=12)
plt.xticks(size=12)
plt.legend(['Train', 'Test'], loc='upper right')
# plt.show()

