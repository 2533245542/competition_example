import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics

#%%
import os
templateDirectoryCommandDict = {'cnnClean': 'run.py'}
tune_interval_list = [1,3,5,7,9,12,15,30]
days_ahead_list = [1,2,3,4,5,6]
repetitions = 5
paths = []
commands = []

#%% grid plot
plot_verbose = False
# i = 1
# a = 1
def make_plot(i, a, ax):
    if plot_verbose:
        df = pd.read_csv('total.csv')[lambda df: df.i == i][lambda df: df.a == a][['testDataTimeStamps', 'list_predict_future', 'list_actual_future', 'list_loss_future', 'list_loss_past']].set_index('testDataTimeStamps')
        df.columns = ['predict', 'actual', 'error of the test day', 'error of in training']
    else:
        df = pd.read_csv('total.csv')[lambda df: df.i == i][lambda df: df.a == a][['testDataTimeStamps', 'list_predict_future', 'list_actual_future']].set_index('testDataTimeStamps')
        df.columns = ['predict', 'actual']

    mae = str(sklearn.metrics.mean_absolute_error(df['predict'], df['actual']))[:4]

    df.plot(ax=ax)
    ax.set_title('tune interval: {} days, \ndays to predict ahead: {}, MAE:{}'.format(i, a, mae))


num_rows = len(tune_interval_list)
num_cols = len(days_ahead_list)
figure, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 30), sharex=True, sharey=True)

for r in range(num_rows):
    for c in range(num_cols):
        make_plot(tune_interval_list[r], days_ahead_list[c], axes[r, c])

plt.savefig('curve_dayAhead_vs_predictionInterval.png')


#%%
# num_rows = len(tune_interval_list)
# num_cols = len(days_ahead_list)
# figure, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 30), sharex=True, sharey=True)
#
# for r in range(num_rows):
#     for c in range(num_cols):
#         make_plot(tune_interval_list[r], days_ahead_list[c], axes[r, c])
#
# # df = pd.read_csv('total.csv')[lambda df: df.i == 30][lambda df: df.a == 1][['testDataTimeStamps', 'list_predict_future', 'list_actual_future']].set_index('testDataTimeStamps')
# df = pd.read_csv('total.csv')[['testDataTimeStamps', 'i', 'a', 'list_predict_future', 'list_actual_future']].set_index('testDataTimeStamps')
# sklearn.metrics.mean_absolute_error(df['predict'], df['actual'])
