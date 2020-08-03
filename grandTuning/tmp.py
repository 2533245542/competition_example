import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('grandTuningResult.csv')

# df.pivot(index='days_ahead', columns='tune_interval', values=['average', 'standardError'])
df_mean = df.pivot(index='tune_interval', columns='days_ahead', values='average')
df_std = df.pivot(index='tune_interval', columns='days_ahead', values='standardError')
#           interval1, 2,...
# days_ahead average mean
#
#           interval1, 2,...
# days_ahead error

#%% mean-error plot, x axis is the number of days between hyperparemter tuning, y is the mean absolute error
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(6,15), sharex=True, sharey=True)

ax = axs[0]
ax.errorbar(y=df_mean.iloc[:, 0], x=df_mean.index, yerr=df_std.iloc[:, 0])
ax.set_title('predicting 1 day ahead')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
for i,j in zip(df_mean.index,df_mean.iloc[:, 0]):
    ax.annotate(str(j)[:3],xy=(i,j))


ax = axs[1]
ax.errorbar(y=df_mean.iloc[:, 1], x=df_mean.index, yerr=df_std.iloc[:, 1])
ax.set_title('predicting 2 day ahead')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
for i,j in zip(df_mean.index,df_mean.iloc[:, 1]):
    ax.annotate(str(j)[:3],xy=(i,j))



ax = axs[2]
ax.errorbar(y=df_mean.iloc[:, 2], x=df_mean.index, yerr=df_std.iloc[:, 2])
ax.set_title('predicting 3 day ahead')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
for i,j in zip(df_mean.index,df_mean.iloc[:, 2]):
    ax.annotate(str(j)[:3],xy=(i,j))



ax = axs[3]
ax.errorbar(y=df_mean.iloc[:, 3], x=df_mean.index, yerr=df_std.iloc[:, 3])
ax.set_title('predicting 4 day ahead')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
for i,j in zip(df_mean.index,df_mean.iloc[:, 3]):
    ax.annotate(str(j)[:3],xy=(i,j))



ax = axs[4]
ax.errorbar(y=df_mean.iloc[:, 4], x=df_mean.index, yerr=df_std.iloc[:, 4])
ax.set_title('predicting 5 day ahead')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
for i,j in zip(df_mean.index,df_mean.iloc[:, 4]):
    ax.annotate(str(j)[:3],xy=(i,j))



ax = axs[5]
ax.errorbar(y=df_mean.iloc[:, 5], x=df_mean.index, yerr=df_std.iloc[:, 5])
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
ax.set_title('predicting 6 day ahead')
for i,j in zip(df_mean.index,df_mean.iloc[:, 5]):
    ax.annotate(str(j)[:3],xy=(i,j))


# plt.show()
plt.savefig('mean-error.png', dpi=200)

#%% mean plot, error plot
fig = plt.figure()
df_mean.plot()
plt.title('plot mean absolute error changing with \ntuning intervals and the number of days to predict ahead ')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('mean absolute error')
plt.legend(title='number of days \nto predict ahead')
# plt.show()
plt.savefig('mean.png')

fig = plt.figure()
df_std.plot()
plt.title('plot mean absolute error instability (standard error)\n changing with tuning intervals and\n the number of days to predict ahead ')
plt.xlabel('number of days between hyperparameter tunings')
plt.ylabel('standard error of mean absolute error\n across 10 evaluations')
plt.legend(title='number of days \nto predict ahead')
# plt.show()
plt.savefig('std.png')


