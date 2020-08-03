import pandas as pd
import matplotlib.pyplot as plt
#%% set date as index
covidDf = pd.read_csv('covid_calls_05062020.csv')
covidDf.columns = ['date', 'clinic', 'call', 'case']


def convert(s):
    return '/'.join([s[:2], s[2:5], s[5:]])


covidDf['date'] = pd.to_datetime(covidDf.date.apply(convert).apply(str), infer_datetime_format=True)
covidDf = covidDf.set_index(['date'])
#%% plot a site
def plot_a_site(site, ax):
    print(site)
    dfSite = covidDf[lambda df: df.clinic == site].resample('D').asfreq().fillna(method='ffill')[['case']]
    dfSite[-130:].plot(ax=ax, title=site)


#%% sites
nonExistingSite = ['612']  # this site has absolutely no record
sites = list(covidDf.clinic.str[:3].unique()[1:])
sites = [site for site in sites if site not in nonExistingSite]

#%% first plot
rows = 5
cols = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
plt.show()


#%% second plot
rows = 5
cols = 6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
plt.show()

#%% third plot
rows = 6
cols = 7
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
plt.show()

#%% fourth plot
rows = 6
cols = 7
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
for r in range(rows):
    for c in range(cols):
        plot_a_site(sites.pop(0), axes[r, c])
plt.show()


#%%
# covidDf[lambda df: df.clinic == '612'].resample('D').asfreq().fillna(method='ffill')[['call', 'case']].plot()

# sites that have similar COVID-19 related case patterns
['436', '438', '463', '501', '504', '506', '508', '512', '515', '516', '526', '528', '531', '537', '539', '541', '544', '546', '548', '549', '550', '553', '554', '556', '561', '573', '590', '608', '613', '621', '623', '626', '631', '632', '642', '644', '646', '650', '658', '659', '662', '663', '668', '673', '674', '678', '679', '688', '695', '756', '635', '693']



