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


#%% sites
nonExistingSite = ['612']  # this site has absolutely no record
sites = list(covidDf.clinic.str[:3].unique()[1:])  # 130 major US regional VA hospitals
sites = [site for site in sites if site not in nonExistingSite]

#%% first plot
rows = 5
cols = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 30))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
# plt.show()
plt.savefig('select1.png')

#%% second plot
rows = 5
cols = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 30))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
# plt.show()
plt.savefig('select2.png')
#%% third plot
rows = 6
cols = 7
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 30))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
# plt.show()
plt.savefig('select3.png')

#%% fourth plot
rows = 6
cols = 7
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 30))
try:
    for r in range(rows):
        for c in range(cols):
            plot_a_site(sites.pop(0), axes[r, c])
# plt.show()
except IndexError:
    plt.savefig('select4.png')

#%% major US regional VA hospitals that have similar COVID-19 related case patterns
['436', '438', '463','504', '506', '508', '516', '526', '528', '531', '537', '539', '541', '544', '546', '548',
         '549', '550', '553', '554', '556', '561', '590', '608', '621', '626', '632', '635', '642', '644', '646', '658', '659',
         '662', '663', '668', '673', '674', '678', '679', '688', '693', '695', '756']

# 515
# 501
# 512
# 573
# 613
# 623
# 631
# 650

