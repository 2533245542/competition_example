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
prev_df = pd.DataFrame({'hospital': ['663', '663A4', '663GA', '663GB', '663GC', '663GD', '663GE'], 'loss': [0.8804780876494024, 1.066147859922179, 1.288888888888889, 1.763265306122449, 1.3206751054852321, 0.8138528138528138, 1.135135135135135]})
curr_df = pd.read_csv(currentFileSaveName,
                      names=['loss', 'hospital'])

# get newest run of each hospital
intermediate_df = curr_df.sort_values('hospital')[['loss', 'hospital']]

full_df = pd.merge(intermediate_df, prev_df, on='hospital', suffixes=('_curr', '_prev'))


# compare with default
diff = (full_df['loss_prev'] / full_df['loss_curr'] - 1)  # positive means improve, negative means degrade
print(pd.DataFrame({'diff': diff, 'site': ['663', '663A4', '663GA', '663GB', '663GC', '663GD', '663GE']}))
print('overall', diff.mean())


