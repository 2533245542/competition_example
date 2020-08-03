# %%
import os

keyword = 'asdfasdfsliding_window_size = '
replacementFormat = 'sliding_window_size = {}\n'
values = [1]

templateDirectoryCommandDict = {'cnnClean': 'run.py'}
repetitions = 10


paths = []
commands = []

def editConfig(path, keyword, replacement):
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if keyword in lines[i]:
                lines[i] = replacement

    with open(path, 'w') as f:
        f.writelines(lines)


for templateDirectory in templateDirectoryCommandDict:
    for value in values:
        valueFolder = '{}_v{}'.format(templateDirectory, value)
        os.system('mkdir {}'.format(valueFolder))
        for repetition in range(repetitions):
            templateDirectory_repetition = '{}_{}'.format(templateDirectory, repetition)
            os.system('cp -r {} {}/{}'.format(templateDirectory, valueFolder, templateDirectory_repetition))
            editConfig('{}/{}/config.py'.format(valueFolder, templateDirectory_repetition),
                       keyword, replacementFormat.format(value))
            commands.append('cd {}/{}; python {} >> output'.format(valueFolder, templateDirectory_repetition, templateDirectoryCommandDict[templateDirectory]))

with open('commandsToRun', 'w') as f:
    commands = [command + '\n' for command in commands]
    f.writelines(commands)
