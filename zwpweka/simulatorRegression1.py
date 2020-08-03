"""
A follow up of simulator 5. We now use min score and regression for early stopping.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def earlyStopWithPatience(algorithm, history, optRd, algorithmToKeep, patience, delta):
    ### weipeng start

    currentScores = history[lambda df: df.algName == algorithm].score

    minScores = currentScores.expanding().min().reset_index(drop=True).tolist()

    #### fit function start
    def func(x, a, b, c, d):
        return a * np.exp(-b * (x-d)) + c  # a, b must be positive

    xdata = np.arange(0, len(minScores), 1)
    ydata = minScores
    popt, _ = curve_fit(func, xdata, ydata, bounds=([0, 0, -np.inf, -np.inf], [+np.inf, 10, +np.inf, +np.inf]), maxfev=10000)  # tune the bound further
    #### fit function end

    #### calculate sigma
    sigma = np.sqrt(np.mean((func(xdata, *popt) - ydata) ** 2))

    #### predict p steps after
    y_p = func(len(minScores) - 1 + patience, *popt)
    stopCondition = ydata[len(ydata) - 1] - (y_p - 2*sigma) < delta

    # print('earlyStopWithPatience', algorithm, sigma, stopCondition, sep=', ')

    if stopCondition:
        return True, history
    else:
        return False, history

    ### weipeng end


def earlyStopWithEndScore(algorithm, history, optRd, num_expected_search_this_round):
    ### weipeng start

    currentScores = history[lambda df: df.algName == algorithm].score

    minScores = currentScores.expanding().min().reset_index(drop=True).tolist()

    #### fit function start
    def func(x, a, b, c, d):
        return a * np.exp(-b * (x-d)) + c  # a, b must be positive

    xdata = np.arange(0, len(minScores), 1)
    ydata = minScores
    popt, _ = curve_fit(func, xdata, ydata, bounds=([0, 0, -np.inf, -np.inf], [+np.inf, 10, +np.inf, +np.inf]), maxfev=10000)  # tune the bound further
    #### fit function end

    #### calculate sigma
    sigma = np.sqrt(np.mean((func(xdata, *popt) - ydata) ** 2))

    #### predict end score of this search
    y_end = func(num_expected_search_this_round - 1, *popt)

    #### find worst algorithm in history
    excluded_algorithms = [algorithm]
    if optRd == 0 or optRd == 1:
        excluded_algorithms = excluded_algorithms + ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO']
    worstScore = history[lambda df: ~df.algName.isin(excluded_algorithms)][['algName', 'score']].groupby('algName', as_index=False).min().sort_values('score', ascending=False).score.tolist()[0]

    #### construct stop condition
    stopCondition = (y_end - 2*sigma) > worstScore

    # print('earlyStopWithEndScore', algorithm, sigma, stopCondition, sep=', ')

    if stopCondition:
        return True, history
    else:
        return False, history


    ### weipeng end

def makeHistoryBetter(algorithm, history, optRd, algorithmToKeep):
    algorithmsInHistoryBestScoreOnTop = history[['algName', 'score']].groupby('algName', as_index=False).min().sort_values('score').algName.tolist()  # history here should contain (algorithmToKeep + algorithm considered for early stopping) algorithms
    algorithmsInHistoryBestScoreOnTop.remove(algorithm)
    numToRemove = len(algorithmsInHistoryBestScoreOnTop) - algorithmToKeep
    if numToRemove < 1:
        return history

    removed_algorithm = []
    for algo in reversed(algorithmsInHistoryBestScoreOnTop):
        if (optRd == 0 or optRd == 1) and algo in ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO']:
            continue
        removed_algorithm.append(algo)
        if len(removed_algorithm) == numToRemove:
            break

    algorithmsKeepInNextHistory = [algo for algo in algorithmsInHistoryBestScoreOnTop if algo not in removed_algorithm]
    algorithmsKeepInNextHistory.append(algorithm)
    history = history[lambda df: df.algName.isin(algorithmsKeepInNextHistory)]

    # if removed_algorithm:
        # print('makeHistoryBetter', algorithmsKeepInNextHistory, len(algorithmsKeepInNextHistory), removed_algorithm, sep=', ')
    return history


def discardFirstRound(algTypeString, optRd, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding 0th round
    algorithmList = cleanDf[lambda df: df.optRd == optRd][['algTypeString', 'algName']].drop_duplicates()
    history = pd.DataFrame(columns=['algName', 'algTypeString', 'optRd', 'score'])
    for i, algorithm in enumerate(algorithmList[lambda df: df.algTypeString == algTypeString].algName):
        # if algorithm == 'weka.classifiers.rules.PART':  # right after OneR
        #     break
        significant = False
        skippedCount = 0
        for j, score in enumerate(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm]['score']):
            if significant:
                skippedCount += 1
                if j == len(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm][
                                'score']) - 1 and skippedCount > 0:
                    print('skip:', algorithm)
                    print('times:', skippedCount)
                continue
            with open('sumulatorDataIter.csv', 'a') as f:
                f.write(','.join([str(i) for i in [algorithm, score]]) + '\n')

            history = history.append(
                {'algName': algorithm, 'algTypeString': algTypeString, 'optRd': optRd, 'score': score},
                ignore_index=True)

            # if algorithm == 'weka.classifiers.rules.OneR':  # right after OneR
            # print(i, len(algorithmList[lambda df: df.algTypeString == algTypeString].algName) * 0.5,
            #       history[lambda df: df.algName == algorithm].__len__(),
            #       history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean() * 0.5)

            algRatio = 0.5  # start after 50% algos are run
            runRatio = 0.5  # start after 50% of lambdas in a algo are run

            currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
            if currentRunNumber > minRequiredRun:
                significant, history = earlyStopWithPatience(algorithm, history, optRd, algorithmToKeep, patience, delta)
                if significant:
                    continue

            if (i+1) > algorithmToKeep: # i+1 is the number of algorithms started in this round. When we first arrive here, we will in history: algorithmToKeep + 1 algorithms. The extra one is the one we go on and test early stop or not
                avgRunNumber = history[lambda df: df.algName != algorithm].groupby(
                    ['algName']).size().mean()  # only counting run number for non-early stopped algorithms
                currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
                if currentRunNumber > minRequiredRun:
                    if algorithm in ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO']:
                        continue
                    # check history here
                    history = makeHistoryBetter(algorithm, history, optRd, algorithmToKeep)
                    # do early stopping based on end score
                    significant, history = earlyStopWithEndScore(algorithm, history, optRd, num_expected_search_this_round)

    return history[['algTypeString', 'algName']].drop_duplicates()  # the left algorithms


def discardOtherRound(optRd, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding other round
    algorithmList = cleanDf[lambda df: df.optRd == optRd][['algTypeString', 'algName']].drop_duplicates()
    history = pd.DataFrame(columns=['algName', 'algTypeString', 'optRd', 'score'])  # optRd would actually not change
    for i, algorithm in enumerate(algorithmList.algName):
        significant = False
        skippedCount = 0
        for j, score in enumerate(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm]['score']):
            if significant:
                skippedCount += 1
                if j == len(cleanDf[lambda df: df.optRd == optRd][lambda df: df.algName == algorithm][
                                'score']) - 1 and skippedCount > 0:
                    print('skip:', algorithm)
                    print('times:', skippedCount)
                continue

            with open('sumulatorDataIter.csv', 'a') as f:
                f.write(','.join([str(i) for i in [algorithm, score]]) + '\n')

            history = history.append({'algName': algorithm, 'algTypeString':
                algorithmList[lambda df: df.algName == algorithm].algTypeString.unique().tolist()[0], 'optRd': optRd,
                                      'score': score}, ignore_index=True)

            # if algorithm == 'weka.classifiers.trees.J48':
            #     print(i, len(algorithmList.algName) * 0.5, history[lambda df: df.algName == algorithm].__len__(), history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean() * 0.5)

            currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
            if currentRunNumber > minRequiredRun:
                significant, history = earlyStopWithPatience(algorithm, history, optRd, algorithmToKeep, patience, delta)
                if significant:
                    continue

            if (i + 1) > algorithmToKeep:  # i+1 is the number of algorithms started in this round. When we first arrive here, we will in history: algorithmToKeep + 1 algorithms. The extra one is the one we go on and test early stop or not
                avgRunNumber = history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean()
                currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
                if currentRunNumber > minRequiredRun:
                    if algorithm in ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO'] and optRd == 1:
                        continue
                    # check history here
                    history = makeHistoryBetter(algorithm, history, optRd, algorithmToKeep)
                    # do early stopping based on end score
                    significant, history = earlyStopWithEndScore(algorithm, history, optRd, num_expected_search_this_round)
    return history[['algTypeString', 'algName']].drop_duplicates()


################################################################################
################################################################################

with open('sumulatorDataIter.csv', 'w') as f:
    pass

df = pd.read_csv('experiments/creditg/out/singleTechnique30.csv',
                 names=['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score', 'subScore', 'time',
                        'argStr'])

cleanDf = df[['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score']].drop_duplicates()[
    ['algName', 'algTypeString', 'optRd', 'i', 'score']]

print('\n', 'next')
discardFirstRound('Base', 0, 8, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30)
# print('\n', 'next')
# print(discardFirstRound('Meta', 0, 4, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30))
# print('\n', 'next')
# discardOtherRound(1, 8, minRequiredRun=15, patience=3, delta=1, num_expected_search_this_round=30)
# print('\n', 'next')
# discardOtherRound(2, 5, minRequiredRun=10, patience=3, delta=1, num_expected_search_this_round=20)
# print('\n', 'next')
# discardOtherRound(3, 3, minRequiredRun=5, patience=3, delta=1, num_expected_search_this_round=10)
