import os
templateDirectoryCommandDict = {'cnnClean': 'run.py'}
tune_interval_list = [1,2,3,4,5,30]
days_ahead_list = [1]
repetitions = 10

for templateDirectory in templateDirectoryCommandDict:
    for tune_interval in tune_interval_list:
        for days_ahead in days_ahead_list:
            for repetition in range(repetitions):
                path = '{}_i{}_a{}/{}_{}'.format(templateDirectory, tune_interval, days_ahead, templateDirectory, repetition)
                print(path)
                os.system('tail -n -1 {}/output'.format(path))
# tail -n 1 */*/output
