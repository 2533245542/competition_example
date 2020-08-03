import pandas as pd

#%%
import os
templateDirectoryCommandDict = {'cnnClean': 'run.py'}
tune_interval_list = [1,3,5,7,9,12,15,30]
days_ahead_list = [1,2,3,4,5,6]
repetitions = 5
paths = []
commands = []

for templateDirectory in templateDirectoryCommandDict:
    for tune_interval in tune_interval_list:
        for days_ahead in days_ahead_list:
            interval_days_ahead_folder = '{}_i{}_a{}'.format(templateDirectory, tune_interval, days_ahead)
            df_repetitions = []
            for repetition in range(repetitions):
                templateDirectory_repetition = '{}_{}'.format(templateDirectory, repetition)
                df_repetitions.append(pd.read_csv('{}/{}/total_prediction_loss_timestamp.csv'.format(interval_days_ahead_folder, templateDirectory_repetition, templateDirectoryCommandDict[templateDirectory]), index_col=0))

            pd.concat(df_repetitions).groupby('testDataTimeStamps', as_index=False).mean().to_csv('{}.csv'.format(interval_days_ahead_folder), index=False)
