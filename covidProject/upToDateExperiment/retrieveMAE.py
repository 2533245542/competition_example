import os
import subprocess

# %% define experiemtns and tuning runs
experiments = [
    'upToDateFeatures2',
    'upToDateFeatures3',
    'upToDateFeatures4',
    'upToDateFeatures5',
    'upToDateFeatures6',
    'upToDateFeatures7',
    'upToDateFeatures8'
]
tuningRuns = [
    'caseOnly_5',
    'caseOnly_10',
    'caseOnly_20',
    'caseOnly_30',
    'caseOnly_40',
    'caseOnly_60',
    'caseOnly_80',
    'caseOnly_100'
]

#%% make commands
commands = []
'tail -n 1 upToDateFeatures7/caseOnly_5/nohup.out'
for experiment in experiments:
    for tuningRun in tuningRuns:
        commands.append('tail -n 1 {}/{}/nohup.out'.format(experiment, tuningRun))

#%% retrieve MAE
experiment_list = []
tuningRun_list = []
MAE_list = []
for experiment in experiments:
    for tuningRun in tuningRuns:
        output = subprocess.check_output(commands.pop(0), shell=True).decode('utf-8').strip()
        print(output.encode('utf-8'))
        # 'MAE_n_day_avg_window_size_100 2.67'

        experiment_list.append(experiment)
        tuningRun_list.append(tuningRun)

        if 'MAE_n_day_avg_window_size_' in output:
            MAE_list.append(output.split(' ')[1])
        else:
            MAE_list.append('')

#%% write output
with open('upToDateExperimentResult.csv', 'w') as f:
    for experiment, tuningRun, MAE in zip(experiment_list, tuningRun_list, MAE_list):
        f.write('{},{},{}\n'.format(experiment, tuningRun, MAE))
