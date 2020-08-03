import subprocess
import multiprocessing
import pandas as pd

command_list = ['python3 663baseLSTMMultivariate.py ../editData/663dataMultivariateLSTM.csv',
                'python3 663A4baseLSTMMultivariate.py ../editData/663A4dataMultivariateLSTM.csv',
                'python3 663GAbaseLSTMMultivariate.py ../editData/663GAdataMultivariateLSTM.csv',
                'python3 663GBbaseLSTMMultivariate.py ../editData/663GBdataMultivariateLSTM.csv',
                'python3 663GCbaseLSTMMultivariate.py ../editData/663GCdataMultivariateLSTM.csv',
                'python3 663GDbaseLSTMMultivariate.py ../editData/663GDdataMultivariateLSTM.csv',
                'python3 663GEbaseLSTMMultivariate.py ../editData/663GEdataMultivariateLSTM.csv']

hospital_index = ['663', '663A4', '663GA', '663GB', '663GC', '663GD', '663GE']
historyFileSaveName = 'history.csv'
currentFileSaveName = 'current.csv'

#%% run models
def func(command, hospital: str, historyFileSaveName, currentFileSaveName):
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', shell=True).stdout.split('\n')
    lock.acquire()
    print(command)
    with open(historyFileSaveName, 'a') as f:
        f.write(output[-2] + ',' + hospital + '\n')
    with open(currentFileSaveName, 'a') as f:
        f.write(output[-2] + ',' + hospital + '\n')

    lock.release()

def init(lock_):
    global lock
    lock = lock_

#%%
with open(currentFileSaveName, 'w') as f:
    pass

lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes=7, initializer=init, initargs=(lock, ))
for i, command in enumerate(command_list):
    pool.apply_async(func, args=(command, hospital_index[i], historyFileSaveName, currentFileSaveName))

pool.close()
pool.join()

#%% make comparison with baseline
prev_df = pd.DataFrame({'hospital': ['663', '663A4', '663GA', '663GB', '663GC', '663GD', '663GE'], 'loss': [1.49, 1.36, 2.30, 2.31, 1.44, 1.14, 1.42]})
curr_df = pd.read_csv(currentFileSaveName,
                      names=['loss', 'hospital'])

# get newest run of each hospital
intermediate_df = curr_df.sort_values('hospital')[['loss', 'hospital']]

full_df = pd.merge(intermediate_df, prev_df, on='hospital', suffixes=('_curr', '_prev'))

# compare with default
diff = (full_df['loss_prev'] / full_df['loss_curr'] - 1)  # positive means improve, negative means degrade
print(pd.DataFrame({'diff': diff, 'site': ['663', '663A4', '663GA', '663GB', '663GC', '663GD', '663GE']}))
print('overall', diff.mean())
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame({'herd immunity threashold': [1 - 1/i for i in np.arange(2.2, 6, 0.1)], 'R0':np.arange(2.2, 6, 0.1)}).plot(x='R0',y='herd immunity threashold')
plt.show()
# #%%
# def calculateDays(initialInfection, totalPopulation, percentageNeeded, R0):
#     totalInfection = initialInfection
#     for i in range(10000):
#         totalInfection = totalInfection * 1.15
#         if totalInfection/totalPopulation > percentageNeeded:
#             break
#     return i
# calculateDays(760, 327000000, 65, 3.5)

# 760 **
# {12: 8000, 6: 2800}
# y = a * np.log(b)
# 8000 = a * np.log(12) + b
# 2800 = a * np.log(6) + b
# (8000-b)/(2800-b) = np.log(12) / np.log(6)
# pd.Series(0.03 * np.log(np.arange(1, 30) + 0)).plot()
# 7:3000
# 12:8000
# 1000/9000
# pd.Series(100 * 1.205 ** np.arange(7,12)).plot()
# plt.show()
