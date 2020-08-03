import os
import multiprocessing
''':arg
'''
cores = 5

commands = []
with open('commandsToRun', 'r') as f:
    commands = [command.strip() for command in f.readlines()]

def f(command):
    print(command)
    os.system(command)

pool = multiprocessing.Pool(processes=cores)

for command in commands:
    pool.apply_async(f, args=(command,))

pool.close()
pool.join()
