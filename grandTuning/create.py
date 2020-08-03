#%%
import os
templateDirectoryCommandDict = {'cnnClean': 'run.py'}
tune_interval_list = [1,3,5,7,9,12,15,30,50,70,90]
days_ahead_list = [1,2,3,4,5,6]
repetitions = 6
paths = []
commands = []

def editConfig(path, tune_interval, days_ahead):
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'tune_frequency = ' in lines[i]:
                lines[i] = 'tune_frequency = {}\n'.format(tune_interval)
            if '''    hp.choice('steps_ahead', ''' in lines[i]:
                lines[i] = '''    hp.choice('steps_ahead', [{}]),\n'''.format(days_ahead)

    with open(path, 'w') as f:
        f.writelines(lines)

for templateDirectory in templateDirectoryCommandDict:
    for tune_interval in tune_interval_list:
        for days_ahead in days_ahead_list:
            interval_days_ahead_folder = '{}_i{}_a{}'.format(templateDirectory, tune_interval, days_ahead)
            os.system('mkdir {}'.format(interval_days_ahead_folder))
            for repetition in range(repetitions):
                templateDirectory_repetition = '{}_{}'.format(templateDirectory, repetition)
                os.system('cp -r {} {}/{}'.format(templateDirectory, interval_days_ahead_folder, templateDirectory_repetition))
                editConfig('{}/{}/config.py'.format(interval_days_ahead_folder, templateDirectory_repetition), tune_interval, days_ahead)
                commands.append('cd {}/{}; python {} >> output'.format(interval_days_ahead_folder, templateDirectory_repetition, templateDirectoryCommandDict[templateDirectory]))

with open('commandsToRun', 'w') as f:
    commands = [command + '\n' for command in commands]
    f.writelines(commands)
