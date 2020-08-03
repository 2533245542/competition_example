# this is a follow-up of earlyStopSimulator and simulatorRegression1.py
import pandas as pd
import os
import time
import numpy as np
from scipy.optimize import curve_fit


def discardFirstRound(algTypeString, optRd, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding 0th round
    algorithmList = cleanDf[lambda df: df.algTypeString == algTypeString][lambda df: df.optRd == optRd][
        ['algTypeString', 'algName']].drop_duplicates()
    for i, algorithm in enumerate(algorithmList[lambda df: df.algTypeString == algTypeString].algName):
        # if algorithm == 'weka.classifiers.rules.PART':  # right after OneR
        #     break
        significant = False
        for j, score in enumerate(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm]['score']):
            with open('experiments/creditg/out/earlyStopGrandHistory.csv', 'a') as f:
                f.write(','.join([time.ctime(), 'creditg', 'Base', algorithm, str(optRd),
                                  str(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm].i.iloc[
                                          j]),
                                  str(score), '123', '123', 'asdfasd asdfasdfa']) + '\n')

            if significant:
                continue
            with open('javaDataIter.csv', 'a') as f:
                f.write(','.join([str(i) for i in [algorithm, score]]) + '\n')

            with open('experiments/creditg/out/earlyStopHistory.csv', 'a') as f:
                f.write(','.join([time.ctime(), 'creditg', 'Base', algorithm, str(optRd),
                                  str(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm].i.iloc[
                                          j]),
                                  str(score), '123', '123', 'asdfasd asdfasdfa']) + '\n')

            exitcode = os.WEXITSTATUS(
                os.system(
                    "python3 techniqueEarlyStopRegression.py /Users/wzhou87/Desktop/zwpweka/experiments/creditg {} Base {} {} {} {} {} {} {} {}".format(
                        algorithm, optRd, algorithmList.__len__(),
                        cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm].i.iloc[j],
                        algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round)
                )
            )
            if exitcode == 20:
                significant = True
                print(exitcode, significant, algorithm, j, score)


def discardOtherRound(optRd, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding other round
    algorithmList = cleanDf[lambda df: df.optRd == optRd][['algTypeString', 'algName']].drop_duplicates()
    for i, algorithm in enumerate(algorithmList.algName):
        # if algorithm == "weka.classifiers.meta.LogitBoost":
        #     break

        significant = False
        for j, score in enumerate(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm]['score']):
            with open('experiments/creditg/out/earlyStopGrandHistory.csv', 'a') as f:
                f.write(','.join([time.ctime(), 'creditg', 'Base', algorithm, str(optRd),
                                  str(cleanDf[lambda df: df.optRd == optRd][
                                          lambda df: df.algName == algorithm].i.iloc[j]),
                                  str(score), '123', '123', 'asdfasd asdfasdfa']) + '\n')
            if significant:
                continue
            with open('javaDataIter.csv', 'a') as f:
                f.write(','.join([str(i) for i in [algorithm, score]]) + '\n')

            with open('experiments/creditg/out/earlyStopHistory.csv', 'a') as f:
                f.write(','.join([time.ctime(), 'creditg', 'Base', algorithm, str(optRd),
                                  str(cleanDf[lambda df: df.optRd == optRd][
                                          lambda df: df.algName == algorithm].i.iloc[
                                          j]),
                                  str(score), '123', '123', 'asdfasd asdfasdfa']) + '\n')
            exitcode = os.WEXITSTATUS(
                os.system(
                    "python3 techniqueEarlyStopRegression.py /Users/wzhou87/Desktop/zwpweka/experiments/creditg {} Base {} {} {} {} {} {} {} {}".format(
                        algorithm, optRd, algorithmList.__len__(),
                        cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm].i.iloc[j],
                        algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round))
            )
            if exitcode == 20:
                significant = True
                print(exitcode, significant, algorithm, j, score)

################################################################################
################################################################################

def init() -> object:
    df = pd.read_csv('experiments/creditg/out/singleTechnique30.csv',
                     names=['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score', 'subScore',
                            'time', 'argStr'])
    with open('experiments/creditg/out/earlyStoppedAlgorithm.csv', 'w') as f:
        pass
    with open('experiments/creditg/out/earlyStopHistory.csv', 'w') as f:
        pass
    with open('experiments/creditg/out/earlyStopGrandHistory.csv', 'w') as f:
        pass
    with open('javaDataIter.csv', 'w') as f:
        pass
    with open('multiclassConditionHistory.csv', 'w') as f:
        pass
    cleanDf = df[['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score']].drop_duplicates()[
        ['algName', 'algTypeString', 'optRd', 'i', 'score']]

    return df, cleanDf


cleanDf: pd.DataFrame
df, cleanDf = init()
print('\n', 'next')
print(discardFirstRound('Base', 0, 8, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30))
print(os.system("python3 calculateSkippedNumber.py experiments/creditg"))
# df, cleanDf = init()
# print('\n', 'next')
# print(discardFirstRound('Meta', 0, 4, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30))
# print(os.system("python3 calculateSkippedNumber.py experiments/creditg"))
# df, cleanDf = init()
# print('\n', 'next')
# print(discardOtherRound(1, 8, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30))
# print(os.system("python3 calculateSkippedNumber.py experiments/creditg"))
# df, cleanDf = init()
# print('\n', 'next')
# print(discardOtherRound(2, 5, minRequiredRun=10, patience=3, delta=1, num_expected_search_this_round=20))
# print(os.system("python3 calculateSkippedNumber.py experiments/creditg"))
# df, cleanDf = init()
# print('\n', 'next')
# print(discardOtherRound(3, 3, minRequiredRun=5, patience=3, delta=1, num_expected_search_this_round=10))
# print(os.system("python3 calculateSkippedNumber.py experiments/creditg"))
