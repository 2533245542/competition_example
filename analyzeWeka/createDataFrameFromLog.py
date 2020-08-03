import os
import pandas as pd
import numpy as np
import re

import automation_script_config as config


"""
Note that this function looks at every log file, regardless of if the experiment is completed or not. 
While making analysis, manual efforts may need to exclude the incomplete experiment


For each .log file, 
    record the folder name
    record the dataset name
    read through each line
        record the round number
        record the round type(base or meta)
        record the argStr
        record the train size, test size, attribute size
        record the time and score
        extract from argStr for algorithm
        make a list of folderName, datasetName, roundNumber, roundType, time, score, algorithm, trainSize, testSize, attributeSize, argStr
"""
def main(zxqwekasName=''):

    # remove Arcene as it is not working
    datasetsNoArcene = [dataset for dataset in config.datasets]  # make a copy of config.datasets
    datasetsNoArcene.pop(2)  # Arcene is datasets[2] and is not working for PSBO

    dataFrameRowAllLogList = []
    # get path name and folder name of each .log file
    for folderName in config.folderNames:  # for each zxqweka folder
        for datasetName in datasetsNoArcene:  # for each experimentFolder in a zxqfolder
            logFilePath = os.path.join(zxqwekasName, folderName, 'experiments', datasetName, 'out',  # make log file path
                                       ''.join([datasetName, '_seed', str(config.seed), '.log']))

            with open(logFilePath) as f:

                dataFrameRowOneLogList = []
                finalTrainScore = None
                finalTestScore = None
                totalUsedTime = None

                for line in f:

                    line = line.strip()

                    '''
                    Base Round 1 search start
                    .... Round . search start
                    Meta Round 1 search start
                    Final Round (5) search completed. 
                    ..... Round... search start 
                    
                    search for 
                    .* search start
                    
                    tokenize the line, use the first token as roundType, the third one as roundNumber, if len(third one) == 3, it means it is the 5th round which is represented as (5)
                    
                    '''

                    if line.endswith('search start...'):
                        splitResult = line.split(' ')
                        roundType = splitResult[0]
                        roundNumber = splitResult[2]
                        if len(roundNumber) == 3:
                            roundNumber = int(roundNumber[1]) # when '(5)'
                        else:
                            roundNumber = int(roundNumber)

                    if line.startswith('Final Round(5) search completed'):
                        roundNumber = 6
                        roundType = 'overall'

                    if line.startswith('argStr: '):
                        algorithm = line.split(' -targetclass ')[1]
                        argStr = line.strip()

                    if line.startswith('Traing & Test set:'):
                        splitResult = line.split(' ')
                        trainSize = int(splitResult[4].split('/')[0])
                        attributeSize = int(splitResult[4].split('/')[1])
                        testSize = int(splitResult[6].split('/')[0])

                    if line.startswith('SubProcessWrapper:'):
                        splitResult = line.split(' ')
                        time = float(splitResult[1][5:-1])
                        score = float(splitResult[2][6:-1])

                        # SubProcessWrapper log is the last message a lambda computation will emit. Add a row only
                        # when it is completed
                        dataFrameRow = [folderName, datasetName, roundNumber, roundType, algorithm, time, score,
                                        trainSize, testSize, attributeSize, argStr]
                        dataFrameRowOneLogList.append(dataFrameRow)

                    if line.startswith('Train errorRate:'):
                        finalTrainScore = float(line.split(': ')[1])

                    if line.startswith('Test errorRate:'):
                        finalTestScore = float(line.split(': ')[1])

                    if line.startswith('Total Used Time:'):
                        totalUsedTime = float(line.split(' ')[2][5:-1])


                # the end of log is reached, add overall info, for each lambda
                for dataFrameRow in dataFrameRowOneLogList:
                    dataFrameRow.extend([finalTrainScore, finalTestScore, totalUsedTime])

                dataFrameRowAllLogList.extend(dataFrameRowOneLogList)

    analysisDataFrame = pd.DataFrame(data=dataFrameRowAllLogList,
                                     columns=['folderName', 'datasetName', 'roundNumber', 'roundType', 'algorithm', 'time', 'score', 'trainSize','testSize', 'attributeSize',  'argStr', 'finalTrainScore', 'finalTestScore', 'totalUsedTime'])

    # 1
    # lets simply do an average across datasets and zxqwekas, excluding datasets that are not finished

    '''
    first we need to remove datasets that are not valid and remove them 
    each invalid dataset, as long as it occurs once in one folder, we remove it from the whole anlysis procedure across all folder
    
    
    So we say the first round has this many algorithms, how many does the first round have?
    How many do the second round have?
    ...
    And the number of algos a round has is defined by the unique 'algorithm', grouped by datasetName, folderName and roundNumber
    for each foldername
        for each datasetName
            for each round, 
                how many unique algorithms are they?
    then we have
    datasetname, foldername, roundnumber, uniqueAlgoritmCount
    
    then we remove datasetname and foldername column, and we havce roundnumber, uniqueAlgorithmCount
    we group by round number, sum up uniqueAlgorithmCount
    and we will have
    round number, uniqueAlgorithmCountSum
    
    we then draw a graph
    '''


    '''
    for each unique folder and unique dataset and round, find the number of algorithms involved
    count the number of algorithms
    consider only rounds and the number of unique algorithms in each round
    sum up the unique algorithm counts in each round
    round num, algorithm, average score, best score
    x axis is algorithms sorted by rounds
    y axis is the best score achieved by an algorithm in the respective round
    
    analysisDataFrame.loc[noIncompleteDatasetsMask, ['roundNumber','algorithm','score']]
    .groupby(['roundNumber','algorithm'], as_index=False) \ 
    .agg({'score': 'max'}) \
    .sort_values('roundNumber', ascending=False) \
    .plot()
    
    4. At the end of each round, what is the best possible(or mean or median) lambda and its score, in one experiemnt?
    x axis is round number
    y axis is 
    
    draw a learning curve for an experiment
    one zxqweka, one dataset
    x axis is the lambda, from highest score to the lowest score, mark the lambda that show up in more than one rounds
    y axis is the score
    we need a df of round, lambda, score
    
    it will also be interesting to mark lambdas by algorithms


    
    '''




    return analysisDataFrame




if __name__ == '__main__':
    main('zxqwekasDiscardOptimization')
