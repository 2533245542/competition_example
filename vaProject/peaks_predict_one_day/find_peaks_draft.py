# import pandas as pd
# potential_peak_range, num_of_peak = 60, 10
#
# #%%
# path = '../editData/with_external_feature_663A4.csv'
# #%%
#
# df = pd.read_csv(path)
# df = df[['n_walkins', 'dayofweek', 'day', 'month', 'year', 'influenza', 'rain', 'snow', 'traffic', 'vacation', 'PRCP',
#          'SNOW', 'TMAX', 'TMIN']]
# df = df[(df['dayofweek'] != 5) & (df['dayofweek'] != 6)]
#
# df_reindex = df.reset_index(drop=True)  # reindex
#
# def get_i_list():
#     df = pd.read_csv(path)
#     df = df[['n_walkins', 'dayofweek', 'day', 'month', 'year', 'influenza', 'rain', 'snow', 'traffic', 'vacation', 'PRCP',
#              'SNOW', 'TMAX', 'TMIN']]
#     df = df[(df['dayofweek'] != 5) & (df['dayofweek'] != 6)]
#     df_reindex = df.reset_index(drop=True)  # reindex
#     # get the index of the deviated points
#     list_of_end_index = (df_reindex.iloc[-potential_peak_range:, 0] - df_reindex.iloc[-potential_peak_range:, 0].mean()).abs().sort_values()[-num_of_peak:].index.tolist()
#     i_list = [len(df_reindex) - 1 - i for i in list_of_end_index]  # the number of units we need to slide to the left such that we are making predictions on the deviated point
#     return i_list
#
#
# (df_reindex.iloc[-potential_peak_range:, 0] - df_reindex.iloc[-potential_peak_range:, 0].mean())[lambda sr: sr > 0].sort_values()
# for i in get_i_list():
#     print(df_reindex[])
#
#
#
# # df_reindex[:len(df_reindex) - (len(df_reindex) - list_of_end_index[0])]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 1))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 52))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 54))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 55))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 24))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 57))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 0))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 13))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 41))]
# df_reindex[:-(len(df_reindex) - (len(df_reindex) - 58))]
#
# # check for correctness
# # mean = df_reindex.iloc[-potential_peak_range:, 0].mean()
# #
# # (df_reindex.iloc[:, 0] - mean).abs()[:978 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:927 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:925 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:924 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:955 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:922 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:979 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:966 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:938 + 1]
# # (df_reindex.iloc[:, 0] - mean).abs()[:921 + 1]
# #
# df_reindex[:978 + 1]
# df_reindex[:927 + 1]
# df_reindex[:925 + 1]
# df_reindex[:924 + 1]
# df_reindex[:955 + 1]
# df_reindex[:922 + 1]
# df_reindex[:979 + 1]
# df_reindex[:966 + 1]
# df_reindex[:938 + 1]
# df_reindex[:921 + 1]
#
#
# # do df is not indexed correctly, we might need to change that
# # how do select df using 1367, such that the last value of the selected df is 1367
# # 1367    1.72
# # 1304    1.72
# # 1334    1.72
# # 1345    1.72
# # 1347    1.72
# # 1370    2.28
# # 1338    2.28
# # 1369    2.28
# # 1353    3.28
# # 1313    3.28
# '''
# for path in paths:
#     find selected index and print it
# '''
# find a way to figure out if the last point we select from the whole dataset has the index we want it to be
#
#
#


#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * np.exp(-b * (x-d)) + c  # a, b must be positive

popt = [1,1,1,1]
xdata = np.arange(0, 16, 0.1)
# plt.plot(xdata+i, func(xdata+i, *popt), 'g--', label='fit: a=%i, b=%i, c=%i, d=%i' % (*popt, ))
# plt.plot([1,2,3,4,5], np.exp(np.array([1,2,3,4,5]) - 3) + 3, 'g--', label='fit: a=%i, b=%i, c=%i, d=%i' % (*popt, ))
# plt.plot([1,2,3,4,5], np.exp(np.array([1,2,3,4,5]) - 4) + 3, 'r--', label='fit: a=%i, b=%i, c=%i, d=%i' % (*popt, ))
plt.legend()
plt.show()
plt.clf()


#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c, d):
    return a * np.exp(-b * (x-d)) + c  # a, b must be positive
    # return a * (b ** (-x)) + c  # a, b must be positive
ydata = np.array([30.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852, 28.856852])
xdata= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata, bounds=([0, 0, -np.inf, -np.inf], [+np.inf, 1, +np.inf, +np.inf]), maxfev=10000)

plt.plot(xdata, func(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % (*popt,))

sigma = np.sqrt(np.mean((func(xdata, *popt) - ydata) ** 2))
# plt.plot(np.arange(-100,100,1), func(np.arange(-100, 100, 1), *popt), 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.legend()
plt.show()
plt.clf()

ydata[len(ydata) - 1] - func(20, *popt) - 2*sigma < delta


np.arange(0, 5, 1)
