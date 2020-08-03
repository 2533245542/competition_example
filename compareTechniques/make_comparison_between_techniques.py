''':arg
To use this script
Make sure the csv is downloaded

eyeball the csv file to see if one dataset has significantly different values, if there is, change it to a regular value
Specify csv file
specify anchor hyperparameter value
specify hyperparameterValues

also change plot axises

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)

# scp wzhou87@mox.hyak.uw.edu:/gscratch/stf/wzhou87/hyperparameterTuning/timeOutSmallDatasetTrain/drawManyCurvesZxqweka0-5/timeOutSmallDatasetTrain.csv ../drawCurve/timeOutSmallDatasetTrain

## change start

#%% hyperparameters
datasetUsed = ['large', 'small', 'all'][1]

files = ['technique42.csv', 'technique42_control.csv']
anchorTechniqueName = 'case'
techniqueNames = ['case', 'control']


ax1_ylim = [-0.02, 0.02]
ax2_ylim = [-0.1, 0.1]
x_axis = files[0].split('.')[0]

## change end


####################################################################################################
#%% overall view

list_of_df = []
for file, techniqueName in zip(files, techniqueNames):
    df_tmp = pd.read_csv(file)
    df_tmp['techniqueName'] = techniqueName
    list_of_df.append(df_tmp)

df = pd.concat(list_of_df).drop(['hyperparameterName', 'hyperparameterValue'], axis=1)

largeDatasets = ['Arcene', 'convex', 'kddcup09appetency', 'dexter', 'mnist', 'mnistrotationbackimagenew', 'amazon', 'gisette', 'cifar10small', 'dorothea', 'cifar10']

if datasetUsed == 'large':
    df = df[lambda df: df.dataset.isin(largeDatasets)]
elif datasetUsed == 'small':
    df = df[lambda df: ~df.dataset.isin(largeDatasets)]
elif datasetUsed == 'all':
    df = df
else:
    exit(1)

print(df.techniqueName.unique())
print(df.testError.isna().groupby([df.techniqueName]).sum().astype(int).reset_index(name='count'))
print('not done:', df.testError.isna().groupby([df.techniqueName]).sum().astype(int).reset_index(name='count').sum().values[1])
print(df.testError.isna().groupby([df.techniqueName, df.dataset]).sum().astype(int).reset_index(name='count'))


#%%
numeOfCompletedDataset = []
errorChangeValueList = []
errorChangeRatioList = []
timeChangeValueList = []
timeChangeRatioList = []

merge_df_list = []  # merge anchor to each other values
for techniqueName in techniqueNames:
    anchorDf = df[lambda df: df.techniqueName == anchorTechniqueName][['dataset', 'testError', 'time']].groupby( 'dataset', as_index=False).mean()
    nonAnchorDf = df[lambda df: df.techniqueName == techniqueName][['dataset', 'testError', 'time']].groupby('dataset', as_index=False).mean()
    merge_df = pd.merge(anchorDf, nonAnchorDf, on='dataset', suffixes=('_this', '_other'))
    numeOfCompletedDataset.append(nonAnchorDf.dropna().__len__())
    # positive means df_this has a larger value than df_other
    merge_df['errorChangeValue'] = merge_df.testError_this - merge_df.testError_other
    merge_df['errorChangeRatio'] = (merge_df.testError_this / merge_df.testError_other - 1)
    merge_df['errorChangeRatio'] = (merge_df.testError_this / merge_df.testError_other - 1).replace([np.inf, -np.inf], np.nan)
    merge_df['timeChangeValue'] = merge_df.time_this - merge_df.time_other
    merge_df['timeChangeRatio'] = merge_df.time_this / merge_df.time_other - 1
    errorChangeValueList.append(merge_df['errorChangeValue'].mean())
    errorChangeRatioList.append(merge_df['errorChangeRatio'].mean())
    timeChangeValueList.append(merge_df['timeChangeValue'].mean())
    timeChangeRatioList.append(merge_df['timeChangeRatio'].mean())
    merge_df_list.append(merge_df)


print('technique names', techniqueNames)
print('number of completed dataset', numeOfCompletedDataset)
print('error change value', errorChangeValueList)
print('error change ratio', errorChangeRatioList)
print('time change value', timeChangeValueList)
print('time change ratio', timeChangeRatioList)


# write a record
if not os.path.exists('result/'):  # experiment folder
    os.mkdir('result')

for techniqueName, merge_df in zip(techniqueNames, merge_df_list):
    merge_df['techniqueName'] = techniqueName

all_concat_df = pd.concat(merge_df_list, axis=0)
all_concat_df.to_csv('result/{}_all_concat_df.csv'.format(x_axis))
print('na numbers are: ')
print(all_concat_df.isnull().sum())
print('\nthe na rows are ')
print(all_concat_df[lambda df: df.isnull().any(axis=1)])
#%% summary
summaryDf = pd.DataFrame(
{'technique names': techniqueNames,
'number of completed dataset': numeOfCompletedDataset,
'overall error rate value change': errorChangeValueList,
'overall error rate percentage change(x100%)': errorChangeRatioList,
'overall time value change': timeChangeValueList,
'overall time percentage change(*100%)': timeChangeRatioList}
)
summaryDf.to_csv('result/{}_summary_df.csv'.format(x_axis))
#%%% make plots

errorRateValueDf = pd.DataFrame({x_axis: techniqueNames, 'overall change value': errorChangeValueList, 'metric': ['overall error rate change value'] * len(techniqueNames)})
errorRateRatioDf = pd.DataFrame({x_axis: techniqueNames, 'overall change ratio': errorChangeRatioList, 'metric': ['overall error rate change ratio'] * len(techniqueNames)})
timeValueDf = pd.DataFrame({x_axis: techniqueNames, 'overall change value': timeChangeValueList, 'metric': ['overall time change value'] * len(techniqueNames)})
timeRatioDf = pd.DataFrame({x_axis: techniqueNames, 'overall change ratio': timeChangeRatioList, 'metric': ['overall time change ratio'] * len(techniqueNames)})
plotDf = pd.concat([errorRateValueDf, errorRateRatioDf, timeValueDf, timeRatioDf], ignore_index=True)

#%% plot individually
fig, axes = plt.subplots(2,2, figsize=(18,10))
errorRateValueDf.plot(x=x_axis, y='overall change value', title='error rate value', ax=axes[0,0])
errorRateRatioDf.plot(x=x_axis, y='overall change ratio', title='error rate ratio', ax=axes[0,1])
timeValueDf.plot(x=x_axis, y='overall change value', title='time value', ax=axes[1,0])
timeRatioDf.plot(x=x_axis, y='overall change ratio', title='time ratio', ax=axes[1,1])
plt.savefig('result/{}_4_plots.png'.format(x_axis), dpi=100)

#%% plot with 2 y axis
pass
#%% new
''':arg
error value: blue
time value: pink
error ratio: purple
time ratio: orange
'''
fig, ax1 = plt.subplots(figsize=(15,7))

## add time change to df so we can plot both
errorRateRatioDf.rename(columns={'overall change ratio': 'error rate'}, inplace=True)
errorRateRatioDf.plot(x=x_axis, ax=ax1, color=['blue'], legend=True)
ax1.set_ylabel('error rate percentage change(x100%)', color='blue')
plt.legend(loc='upper left')  # make legend not overlap

## add time change to df so we can plot both
ax2 = plt.twinx()
timeRatioDf.rename(columns={'overall change ratio': 'time'}, inplace=True)
timeRatioDf.plot(x=x_axis, ax=ax2, color=['orange'], legend=True)
ax2.set_ylabel('time percentage change(x100%)', color='orange')
plt.legend(loc='upper right')  # make legend not overlap


## beautify

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

ax1.set_ylim(ax1_ylim)
ax2.set_ylim(ax2_ylim)

align_yaxis(ax1, 0, ax2, 0) # align y = 0
plt.savefig('result/{}_aligned.png'.format(x_axis), dpi=100)
plt.show()


#%% tmp
# scp wzhou87@mox.hyak.uw.edu:/gscratch/stf/wzhou87/hyperparameterTuning/ratioScoreUpperBound/drawManyCurvesZxqweka0-5/ratioScoreUpperBound.csv ../drawCurve/ratioScoreUpperBound

#%%

