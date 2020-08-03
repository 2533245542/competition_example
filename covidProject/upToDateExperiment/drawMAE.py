#
# x axis is tune frequnency
# y axiss is MAE
#
# legend is number of days to predict ahead
#
# use seabox.factor plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%% data cleaning
df = pd.read_csv('upToDateExperimentResult.csv', names=['experiment', 'tuningRun', 'MAE'])
df['experiment'] = df['experiment'].str[16:].apply(int)
df['tuningRun'] = df['tuningRun'].str[9:].apply(int)
df['MAE'] = df['MAE'].apply(float)

#%% plot
df[lambda df: df.experiment == 2].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 2].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 3].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 3].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 4].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 4].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 5].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 5].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 6].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 6].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 7].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 7].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()
df[lambda df: df.experiment == 8].plot.line(x='tuningRun', y='MAE', legend=None, title='predict {} days ahead'.format(df[lambda df: df.experiment == 8].experiment.unique()[0])); plt.xlabel('interval between performing automated hyperparameter tuning (days)'); plt.ylabel('mean absolute error'); plt.show()


#%%
'x axis should start from 0, and the interval should be 10'