import os
import multiprocessing
import itertools

# file = '663A4baseLSTMMultivariate.py'
# file = '663GAbaseLSTMMultivariate.py'
# file = '663GBbaseLSTMMultivariate.py'
# file = '663GCbaseLSTMMultivariate.py'
# file = '663GDbaseLSTMMultivariate.py'
# file = '663GEbaseLSTMMultivariate.py'
file = '663baseLSTMMultivariate.py'

clinic_name = file.split('base')[0]


def run(file, features):
    # os.system('python3 -W ignore ' + file + ' ' + features + ' >> tmp')
    os.system('python3 -W ignore ' + file + ' ' + features + ' >> combination_record_{}'.format(clinic_name))


cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

date_feature_index = '0, 1'
external_feature_index = [5, 6, 7, 8, 9, 10, 11, 12, 13]
# for i, tuple in enumerate(list(itertools.combinations(external_feature_index, r))):
r = -1
for _ in range(10):
    for i, tuple in enumerate(
            [[7, 11, 13],
             [5, 6, 7, 9, 11, 12, 13],
             [5, 6, 7, 10, 13],
             [5, 7, 9, 12, 13],
             [5, 6, 8, 9, 13],
             [5, 6, 11, 12, 13],
             [5, 6, 8, 9, 10, 12, 13],
             [6, 7, 10, 11, 12],
             [7, 10, 13],
             [5, 6, 8, 10, 13],
             [8, 9, 12, 13],
             [6, 9, 11, 12],
             [5, 6, 7, 9]]
    ):
        print(i, tuple)

        f = open(file, 'r')  # read file
        lines = f.readlines()  # read file
        f.close()

        lines[51] = "df = df.iloc[:, [{}{}]]\n".format(date_feature_index, ''.join([', ' + str(i) for i in tuple]))

        newFile = file.split('.')[0] + str(r) + '_' + str(i) + '.' + file.split('.')[1]
        f = open(newFile, 'w')
        f.writelines(lines)
        f.close()
        pool.apply_async(run, args=(newFile, 'feature_is_{}'.format(','.join([str(i) for i in tuple])),))

pool.close()
pool.join()

# %%
import pandas as pd

print(pd.read_csv('combination_record_{}'.format(clinic_name), sep=' ',
            names=['clinic', 'metric', 'value', 'feature']).sort_values('value')[['value', 'feature']].groupby(
    'feature').mean().sort_values('value')
)

"""
To use this file, do the following:

change 
    file = ...
to the specific model

change  
    date_feature_index = '0, 1, 2'
to the date_feature_index used by the specific model

change the list of tuples in enumerate()  ** remember, use list for a tuple instead of ()
    for i, tuple in enumerate(
    )
"""

