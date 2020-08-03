""" follow up of techniqueEarlyStop.py """

import sys
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


"""
Remove algorithms in history to make it better.
However, removal stops if the history is about to have fewer algorithms that this round plans to keep
"""
def makeHistoryBetter(algorithm, history, optRd, trialI, algorithmToKeep):
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

    if removed_algorithm:
        with open(experimentPath + '/out/earlyStoppedAlgorithm.csv', 'a') as f:
            for s in removed_algorithm:
                f.write(','.join([str(i) for i in [s, optRd, trialI, 'madeBetter']]) + '\n')
    return history


def discardFirstRoundOneAlgorithm(grand, history, algorithm, algTypeString, optRd,
                                  numberCandidateAlgorithm, trialI, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding 0th round
    """ for first round, history is filtered by algTypeString and optRd; for other round, history filtered by optRd only """
    history = history[lambda df: df.optRd == optRd][lambda df: df.algTypeString == algTypeString]
    grand = grand[lambda df: df.optRd == optRd][lambda df: df.algTypeString == algTypeString]
    # cannot infer nor numTestAlgorithm or avgRunNumber from history, as history is always pruned, so we created a new one called grandHistory
    numTestedAlgorithm = grand[lambda df: df.algName != algorithm].algName.unique().__len__()  # number of algorithms ran, excluding current algo
    significant = False
    # if algorithm == 'weka.classifiers.rules.OneR':  # right after OneR
    # print(numTestedAlgorithm, numberCandidateAlgorithm * 0.5, history[lambda df: df.algName == algorithm].__len__(),
    #       history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean() * 0.5)
    currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
    if currentRunNumber > minRequiredRun:
        significant, history = earlyStopWithPatience(algorithm, history, optRd, algorithmToKeep, patience, delta)

    if numTestedAlgorithm > algorithmToKeep:
        avgRunNumber = history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean()
        currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
        if currentRunNumber > minRequiredRun:
            if algorithm in ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO']:
                significant = False
                return significant, history
            history = makeHistoryBetter(algorithm, history, optRd, trialI, algorithmToKeep)  # check history here
            if not significant:
                significant, history = earlyStopWithEndScore(algorithm, history, optRd, num_expected_search_this_round)  # do second early stopping
    if significant:
        with open(experimentPath + '/out/earlyStoppedAlgorithm.csv', 'a') as f:
            f.write(','.join([str(i) for i in [algorithm, optRd, trialI, 'earlyStopped']]) + '\n')
    return significant, history


def discardOtherRoundOneAlgorithm(grand, history, algorithm, optRd, numberCandidateAlgorithm, trialI, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round):  # for discarding other round
    history = history[lambda df: df.optRd == optRd]
    grand = grand[lambda df: df.optRd == optRd]
    numTestedAlgorithm = grand[
        lambda
            df: df.algName != algorithm].algName.unique().__len__()  # number of algorithms ran, excluding current algo
    significant = False

    currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
    if currentRunNumber > minRequiredRun:
        significant, history = earlyStopWithPatience(algorithm, history, optRd, algorithmToKeep, patience, delta)

    if numTestedAlgorithm > algorithmToKeep:
        with open('multiclassConditionHistory.csv', 'a') as f:
            f.write(','.join([str(s) for s in [optRd, numTestedAlgorithm, numberCandidateAlgorithm * 0.5,
                                               history[lambda df: df.algName == algorithm].__len__(),
                                               history[lambda df: df.algName != algorithm].groupby(
                                                   ['algName']).size().mean() * 0.5]]) + '\n')

        avgRunNumber = history[lambda df: df.algName != algorithm].groupby(['algName']).size().mean()
        currentRunNumber = history[lambda df: df.algName == algorithm].__len__()
        if currentRunNumber > minRequiredRun:
            if algorithm in ['weka.classifiers.trees.RandomForest', 'weka.classifiers.functions.SMO'] and optRd == 1:
                significant = False
                return significant, history
            history = makeHistoryBetter(algorithm, history, optRd, trialI, algorithmToKeep)  # check history here
            if not significant:
                significant, history = earlyStopWithEndScore(algorithm, history, optRd)  # do second early stopping
    if significant:
        with open(experimentPath + '/out/earlyStoppedAlgorithm.csv', 'a') as f:
            f.write(','.join([str(i) for i in [algorithm, optRd, trialI, 'earlyStopped']]) + '\n')
    return significant, history


################################################################################
################################################################################

algoRatio = 0.5
runRatio = 0.5
experimentPath = sys.argv[1]
algorithm = sys.argv[2]
algorithmType = sys.argv[3]  # Base
currentOptRd = int(sys.argv[4])  # 0
numberCandidateAlgorithm = int(sys.argv[5])  # e.g. for base round 0,
trialI = int(sys.argv[6])  # 1
algorithmToKeep = int(sys.argv[7])  # 1
minRequiredRun = int(sys.argv[8])
patience = int(sys.argv[9])
delta = float(sys.argv[10])
num_expected_search_this_round = int(sys.argv[11])

# this number is all algorithms considered for base round 0
all = pd.read_csv(experimentPath + '/out/earlyStopHistory.csv',  # read the record file
                  names=['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score', 'subScore',
                         'time', 'argStr'])

history = all[['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score']].drop_duplicates()[
    ['algName', 'algTypeString', 'optRd', 'i', 'score']]  # only consider one score per lambda

grand = pd.read_csv(experimentPath + '/out/earlyStopGrandHistory.csv',  # read the grand history file
                    names=['identifier', 'dataset', 'algTypeString', 'algName', 'optRd', 'i', 'score', 'subScore',
                           'time', 'argStr'])

if currentOptRd == 0:
    significant, history = discardFirstRoundOneAlgorithm(grand, history, algorithm, algorithmType, currentOptRd,
                                                         numberCandidateAlgorithm, trialI, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round)
else:
    significant, history = discardOtherRoundOneAlgorithm(grand, history, algorithm, currentOptRd,
                                                         numberCandidateAlgorithm, trialI, algorithmToKeep, minRequiredRun, patience, delta, num_expected_search_this_round)

all.merge(history[['algName']].drop_duplicates(), on='algName', how='right').to_csv(
    path_or_buf=experimentPath + '/out/earlyStopHistory.csv', header=False, index=False)
# filter all, using history.algName, and write the resulted one to earlyStopHistory.csv

if significant:
    print(sys.argv)
    exit(20)  # significant
else:
    exit(10)  # not significant
