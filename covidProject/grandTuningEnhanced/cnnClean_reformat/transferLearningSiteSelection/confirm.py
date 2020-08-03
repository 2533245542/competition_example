import pandas as pd
import matplotlib.pyplot as plt
#%% set date as index
covidDf = pd.read_csv('../covid_calls_05062020.csv')
covidDf.columns = ['date', 'clinic', 'call', 'case']


def convert(s):
    return '/'.join([s[:2], s[2:5], s[5:]])


covidDf['date'] = pd.to_datetime(covidDf.date.apply(convert).apply(str), infer_datetime_format=True)
covidDf = covidDf.set_index(['date'])
#%% plot a site
def plot_a_site(site, ax):
    print(site)
    dfSite = covidDf[lambda df: df.clinic == site].resample('D').asfreq().fillna(method='ffill')[['case']]
    dfSite[-130:].plot(ax=ax, title='Hospital {}'.format(site))
    ax.set_ylabel('number of COVID-19\nrelated phone call inqueries')


#%% sites that have similar COVID-19 related case patterns
sites = ['436', '438', '463','504', '506', '508', '516', '526', '528', '531', '537', '539', '541', '544', '546', '548',
         '549', '550', '553', '554', '556', '561', '590', '608', '621', '626', '632', '635', '642', '644', '646', '658', '659',
         '662', '663', '668', '673', '674', '678', '679', '688', '693', '695', '756']  # 44 sites
#%% plot sites
rows = 7
cols = 7
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(35, 20), sharex=True)
try:
    for r in range(rows):
        for c in range(cols):
            plot_a_site(sites.pop(0), axes[r, c])
except IndexError:
    # plt.show()
    plt.savefig('confirm.png')



