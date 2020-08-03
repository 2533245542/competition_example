import numpy as np
import subprocess

'==> lstmClean_i1_a1/lstmClean_0/output <=='
def getIndicator(line):
    indicator = line.split(' ')[1].split('/')[0]
    return indicator

'MAE_n_day_avg_window_size_30 2.4005176439190716'
def getError(line):
    if 'MAE_n_day_avg_window_size' in line:
        return float(line.split(' ')[1])
    else:
        return None

def getCmdResult(command):
    return subprocess.check_output(command, shell=True).decode('utf-8').split('\n')

outputList = getCmdResult('tail -n 1 */*/output')
indicatorErrorDict = {}
indicator = None
i = 0

while i < len(outputList):
    indicator = getIndicator(outputList[i])
    i += 1
    if indicator not in indicatorErrorDict.keys():
        indicatorErrorDict[indicator] = []
    error = getError(outputList[i])
    indicatorErrorDict[indicator].append(error)
    i += 1
    i += 1

for indicator in indicatorErrorDict:
    print(indicator, np.mean(indicatorErrorDict[indicator]), np.std(indicatorErrorDict[indicator]), indicatorErrorDict[indicator])
