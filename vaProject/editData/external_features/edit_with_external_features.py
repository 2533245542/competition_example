# %% this file assume a working directory of external_features/
# %% read in data
import pandas as pd
import glob
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from functools import reduce


# %% get date feature for each clinic

def selector_sta6a(Sta3n, sta6a):
    """Different from the one in edit.py, this function does not drop date. It returns the dataframe instead of
    writing it.
    """
    healthCareSystem = Sta3n
    hospitalName = sta6a
    df_selected = pd.read_csv('../walk_in_data.csv')

    # clean data
    df_selected['Sta3n'] = df_selected['Sta3n'].apply(str)
    df_selected['date'] = pd.to_datetime(df_selected['date'])

    # set index
    df_selected = df_selected.set_index(['date'], drop=True)

    # select  healthCareSystem and hospitalName
    df_selected = df_selected[df_selected['Sta3n'] == healthCareSystem][
        lambda df: df['sta6a'] == hospitalName].resample('D').asfreq().fillna(0)
    df_selected = df_selected[['n_walkins']]

    # add attributes
    def addTimeAttributes(df: pd.DataFrame):
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['is_month_end'] = df.index.is_month_end
        df['is_month_start'] = df.index.is_month_start
        df['is_quarter_end'] = df.index.is_quarter_end
        df['is_quarter_start'] = df.index.is_quarter_start
        df['is_year_end'] = df.index.is_year_end
        df['is_year_start'] = df.index.is_year_start
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['week'] = df.index.week
        df['weekday'] = df.index.weekday
        df['weekofyear'] = df.index.weekofyear
        df['year'] = df.index.year
        return df

    df_selected = addTimeAttributes(df_selected)

    # convert date index to int index
    df_selected = df_selected.reset_index(drop=False)

    # convert boolean to int
    df_selected.is_month_end = df_selected.is_month_end.apply(int)
    df_selected.is_month_start = df_selected.is_month_start.apply(int)
    df_selected.is_quarter_end = df_selected.is_quarter_end.apply(int)
    df_selected.is_quarter_start = df_selected.is_quarter_start.apply(int)
    df_selected.is_year_end = df_selected.is_year_end.apply(int)
    df_selected.is_year_start = df_selected.is_year_start.apply(int)

    # # write to csv
    # pathHospital = '/Users/wzhou87/Desktop/vaProject/editData/{}dataMultivariateLSTM.csv'.format(hospitalName)
    # df_selected.to_csv(pathHospital, index=False)
    #
    # pd.read_csv(pathHospital).plot.scatter(x='dayofweek', y='n_walkins')
    # plt.show()
    return df_selected


date_663 = selector_sta6a('663', '663')
date_663A4 = selector_sta6a('663', '663A4')
date_663GA = selector_sta6a('663', '663GA')
date_663GB = selector_sta6a('663', '663GB')
date_663GC = selector_sta6a('663', '663GC')
date_663GD = selector_sta6a('663', '663GD')
date_663GE = selector_sta6a('663', '663GE')
date_663GF = selector_sta6a('663', '663GF')  # not existed


# %% prepare google features
def read_and_resample_and_bfil(fileName, attributeName, skipRows):
    """read the file skipping the first skipRows rows, name the columns as ['date', attributeName]"""
    df_selected = pd.read_csv(fileName, skiprows=skipRows, names=['date', attributeName])
    df_selected['date'] = pd.to_datetime(df_selected['date'])
    # set index
    df_selected = df_selected.set_index(['date'], drop=True)
    return df_selected.resample('D').asfreq().bfill()


df_influenza = read_and_resample_and_bfil(fileName='Influenza_Disease.csv', attributeName='influenza', skipRows=3)
df_rain = read_and_resample_and_bfil(fileName='Rain_topic.csv', attributeName='rain', skipRows=3)
df_snow = read_and_resample_and_bfil(fileName='Snow_topic.csv', attributeName='snow', skipRows=3)
df_traffic = read_and_resample_and_bfil(fileName='Traffic_congestion_topic.csv', attributeName='traffic', skipRows=3)
df_vacation = read_and_resample_and_bfil(fileName='Vacation_topic.csv', attributeName='vacation', skipRows=3)

df_google = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
                   [df_influenza, df_rain, df_snow, df_traffic, df_vacation])

# %% read the whole weather dataset
climate_part_files = glob.glob('climate_wa_*.csv')
climate_df = pd.concat([pd.read_csv(f) for f in climate_part_files], axis=0, ignore_index=True)
# find stations that record temperatures
# climate_df[lambda df: ~df[['TMAX', 'TMIN']].isna().any(axis=1)][['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates().to_clipboard()
# climate_df_subset = climate_df[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'DATE','PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']]
climate_df_subset = climate_df[['NAME', 'DATE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']]
climate_df_subset['DATE'] = pd.to_datetime(climate_df_subset['DATE'])

# %%
# next we select 'precipitation' and 'snow' features for each clinic
climate_663GC = climate_df_subset[lambda df: df.NAME.isin(
    ['MOUNT VERNON 0.6 N, WA US', 'MOUNT VERNON 1.1 E, WA US', 'MOUNT VERNON 0.8 SW, WA US',
     'MOUNT VERNON 1.8 SW, WA US'])].sort_values(['DATE', 'PRCP'], ascending=[True, False])[['DATE', 'PRCP', 'SNOW']]
climate_663GE = climate_df_subset[lambda df: df.NAME.isin(
    ['PORT ANGELES FAIRCHILD INTERNATIONAL AIRPORT, WA US', 'PORT ANGELES 6.0 S, WA US', 'PORT ANGELES 6.1 SSE, WA US',
     'PORT ANGELES 6.7 SSE, WA US', 'PORT ANGELES 8.2 SE, WA US'])].sort_values(['DATE', 'PRCP'],
                                                                                ascending=[True, False])[
    ['DATE', 'PRCP', 'SNOW']]
climate_663GB = climate_df_subset[lambda df: df.NAME.isin(
    ['POULSBO 3.7 NNW, WA US', 'SUQUAMISH 1.4 E, WA US', 'BREMERTON 2.8 NE, WA US', 'TRACYTON 2.3 WSW, WA US',
     'BREMERTON 1.8 NE, WA US', 'BREMERTON 2.9 NNE, WA US', 'BREMERTON 2.9 NNE, WA US'])].sort_values(['DATE', 'PRCP'],
                                                                                                      ascending=[True,
                                                                                                                 False])[
    ['DATE', 'PRCP', 'SNOW']]
climate_663 = climate_df_subset[lambda df: df.NAME.isin(
    ['SEATTLE 5.8 SSE, WA US', 'SEATTLE 5.1 SE, WA US', 'SEATTLE BOEING FIELD, WA US'])].sort_values(['DATE', 'PRCP'],
                                                                                                     ascending=[True,
                                                                                                                False])[
    ['DATE', 'PRCP', 'SNOW']]
climate_663GA = \
    climate_df_subset[lambda df: df.NAME.isin(['BELLEVUE 1.8 W, WA US', 'BELLEVUE 0.8 S, WA US'])].sort_values(
        ['DATE', 'PRCP'], ascending=[True, False])[
        ['DATE', 'PRCP', 'SNOW']]
climate_663A4 = climate_df_subset[lambda df: df.NAME.isin(
    ['STEILACOOM 0.4 NW, WA US', 'PARKLAND 1.0 WSW, WA US', 'PARKLAND 0.9 NE, WA US',
     'SUMMIT 1.1 WSW, WA US'])].sort_values(['DATE', 'PRCP'], ascending=[True, False])[['DATE', 'PRCP', 'SNOW']]
climate_663GD = \
    climate_df_subset[lambda df: df.NAME.isin(['CENTRALIA, WA US', 'CENTRALIA 1.6 W, WA US'])].sort_values(
        ['DATE', 'PRCP'], ascending=[True, False])[
        ['DATE', 'PRCP', 'SNOW']]

# fill na for precipitation
climate_663GC.PRCP.ffill(inplace=True)
climate_663GE.PRCP.ffill(inplace=True)
climate_663GB.PRCP.ffill(inplace=True)
climate_663.PRCP.ffill(inplace=True)
climate_663GA.PRCP.ffill(inplace=True)
climate_663A4.PRCP.ffill(inplace=True)
climate_663GD.PRCP.ffill(inplace=True)

# fill na for snow
climate_663GC.SNOW.fillna(0, inplace=True)
climate_663GE.SNOW.fillna(0, inplace=True)
climate_663GB.SNOW.fillna(0, inplace=True)
climate_663.SNOW.fillna(0, inplace=True)
climate_663GA.SNOW.fillna(0, inplace=True)
climate_663A4.SNOW.fillna(0, inplace=True)
climate_663GD.SNOW.fillna(0, inplace=True)

# print(climate_663GC.SNOW.count(), len(climate_663GC)) # non-na count: 1668, df length: 3309
# print(climate_663GE.SNOW.count(), len(climate_663GE)) # non-na count: 3324, df length: 7904
# print(climate_663GB.SNOW.count(), len(climate_663GB)) # non-na count: 5533, df length: 7621
# print(climate_663.SNOW.count(), len(climate_663)) # non-na count: 1113, df length: 4562
# print(climate_663GA.SNOW.count(), len(climate_663GA)) # non-na count: 1179, df length: 2298
# print(climate_663A4.SNOW.count(), len(climate_663A4)) # non-na count: 2862, df length: 4938
# print(climate_663GD.SNOW.count(), len(climate_663GD)) # non-na count: 23, df length: 2121

# print(climate_663GC.PRCP.count(), len(climate_663GC))  # non-na count: 3133, df length: 3309
# print(climate_663GE.PRCP.count(), len(climate_663GE))  # non-na count: 7767, df length: 7904
# print(climate_663GB.PRCP.count(), len(climate_663GB))  # non-na count: 7575, df length: 7621
# print(climate_663.PRCP.count(), len(climate_663))  # non-na count: 4401, df length: 4562
# print(climate_663GA.PRCP.count(), len(climate_663GA))  # non-na count: 2283, df length: 2298
# print(climate_663A4.PRCP.count(), len(climate_663A4))  # non-na count: 4733, df length: 4938
# print(climate_663GD.PRCP.count(), len(climate_663GD))  # non-na count: 2120, df length: 2121

# group by date and take average of PRCP and SNOW
climate_663GC_per_day_prcp_snow = climate_663GC.groupby('DATE').mean()
climate_663GE_per_day_prcp_snow = climate_663GE.groupby('DATE').mean()
climate_663GB_per_day_prcp_snow = climate_663GB.groupby('DATE').mean()
climate_663_per_day_prcp_snow = climate_663.groupby('DATE').mean()
climate_663GA_per_day_prcp_snow = climate_663GA.groupby('DATE').mean()
climate_663A4_per_day_prcp_snow = climate_663A4.groupby('DATE').mean()
climate_663GD_per_day_prcp_snow = climate_663GD.groupby('DATE').mean()

# print(len(climate_663GC_per_day_prcp_snow))  # 1826
# print(len(climate_663GE_per_day_prcp_snow))  # 1826
# print(len(climate_663GB_per_day_prcp_snow))  # 1826
# print(len(climate_663_per_day_prcp_snow))  # 1826
# print(len(climate_663GA_per_day_prcp_snow))  # 1826
# print(len(climate_663A4_per_day_prcp_snow))  # 1826
# print(len(climate_663GD_per_day_prcp_snow))  # 1826
#
# print(len(climate_663GC_per_day_prcp_snow.resample('D').asfreq()))  # 1697
# print(len(climate_663GE_per_day_prcp_snow.resample('D').asfreq()))  # 1826
# print(len(climate_663GB_per_day_prcp_snow.resample('D').asfreq()))  # 1826
# print(len(climate_663_per_day_prcp_snow.resample('D').asfreq()))  # 1826
# print(len(climate_663GA_per_day_prcp_snow.resample('D').asfreq()))  # 1800
# print(len(climate_663A4_per_day_prcp_snow.resample('D').asfreq()))  # 1823
# print(len(climate_663GD_per_day_prcp_snow.resample('D').asfreq()))  # 1826

# if a pair of lengths are the same, then it means the date is continuous and it is good. But it is not, we see that
# 663GC, 663GA, and 663A4 are not continuous

# some stations do not have continous date, we solve this problem here
climate_663GC_per_day_prcp_snow = climate_663GC_per_day_prcp_snow.resample('D').asfreq()
climate_663GE_per_day_prcp_snow = climate_663GE_per_day_prcp_snow.resample('D').asfreq()
climate_663GB_per_day_prcp_snow = climate_663GB_per_day_prcp_snow.resample('D').asfreq()
climate_663_per_day_prcp_snow = climate_663_per_day_prcp_snow.resample('D').asfreq()
climate_663GA_per_day_prcp_snow = climate_663GA_per_day_prcp_snow.resample('D').asfreq()
climate_663A4_per_day_prcp_snow = climate_663A4_per_day_prcp_snow.resample('D').asfreq()
climate_663GD_per_day_prcp_snow = climate_663GD_per_day_prcp_snow.resample('D').asfreq()

climate_663GC_per_day_prcp_snow.ffill(inplace=True)
climate_663GE_per_day_prcp_snow.ffill(inplace=True)
climate_663GB_per_day_prcp_snow.ffill(inplace=True)
climate_663_per_day_prcp_snow.ffill(inplace=True)
climate_663GA_per_day_prcp_snow.ffill(inplace=True)
climate_663A4_per_day_prcp_snow.ffill(inplace=True)
climate_663GD_per_day_prcp_snow.ffill(inplace=True)

# %% next we select 'max temp' and 'min temp' features for each clinic
temp_663GC = climate_df_subset[lambda df: df.NAME.isin(['SEDRO WOOLLEY, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]
temp_663GE = climate_df_subset[lambda df: df.NAME.isin(['PORT ANGELES FAIRCHILD INTERNATIONAL AIRPORT, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]
temp_663GB = climate_df_subset[lambda df: df.NAME.isin(['BREMERTON, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]
temp_663 = climate_df_subset[lambda df: df.NAME.isin(['BREMERTON, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]
temp_663GA = climate_df_subset[lambda df: df.NAME.isin(['RENTON MUNICIPAL AIRPORT, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]
temp_663A4 = climate_df_subset[lambda df: df.NAME.isin(['TACOMA NUMBER 1, WA US', 'MCMILLIN RESERVOIR, WA US'])].sort_values(['DATE'])[['DATE', 'TMAX', 'TMIN']]
temp_663GD = climate_df_subset[lambda df: df.NAME.isin(['CENTRALIA, WA US'])].sort_values(['DATE', 'TMAX', 'TMIN'])[['DATE', 'TMAX', 'TMIN']]

# print(temp_663GC.count()) # DATE 1826, TMAX 1796, TMIN 1608
# print(temp_663GE.count()) # DATE 1822, TMAX 1808, TMIN 1808
# print(temp_663GB.count()) # DATE 1809, TMAX 1806, MIN 1804
# print(temp_663.count()) # DATE 1809, TMAX 1806, MIN 1804
# print(temp_663GA.count()) # DATE 1822, TMAX 1811, MIN 1811
# print(temp_663A4.count()) # DATE 3044, TMAX 3040, TMIN 3038
# print(temp_663GD.count()) # DATE 1826, TMAX 1826, TMIN 1826

# filling TMAX and TMIN
temp_663GC.ffill(inplace=True)
temp_663GE.ffill(inplace=True)
temp_663GB.ffill(inplace=True)
temp_663.ffill(inplace=True)
temp_663GA.ffill(inplace=True)
temp_663A4.ffill(inplace=True)
temp_663GD.ffill(inplace=True)


temp_663GC_per_day_tmax_tmin = temp_663GC.groupby('DATE').mean()
temp_663GE_per_day_tmax_tmin = temp_663GE.groupby('DATE').mean()
temp_663GB_per_day_tmax_tmin = temp_663GB.groupby('DATE').mean()
temp_663_per_day_tmax_tmin = temp_663.groupby('DATE').mean()
temp_663GA_per_day_tmax_tmin = temp_663GA.groupby('DATE').mean()
temp_663A4_per_day_tmax_tmin = temp_663A4.groupby('DATE').mean()
temp_663GD_per_day_tmax_tmin = temp_663GD.groupby('DATE').mean()

# next figure out the problem of continouslity

# print(len(temp_663GC_per_day_tmax_tmin), len(temp_663GC_per_day_tmax_tmin.resample('D').asfreq()))  # 1826, 1826
# print(len(temp_663GE_per_day_tmax_tmin), len(temp_663GE_per_day_tmax_tmin.resample('D').asfreq()))  # 1822, 1825
# print(len(temp_663GB_per_day_tmax_tmin), len(temp_663GB_per_day_tmax_tmin.resample('D').asfreq()))  # 1809, 1826
# print(len(temp_663_per_day_tmax_tmin), len(temp_663_per_day_tmax_tmin.resample('D').asfreq()))  # 1809, 1826
# print(len(temp_663GA_per_day_tmax_tmin), len(temp_663GA_per_day_tmax_tmin.resample('D').asfreq()))  # 1822, 1825
# print(len(temp_663A4_per_day_tmax_tmin), len(temp_663A4_per_day_tmax_tmin.resample('D').asfreq()))  # 1826, 1826
# print(len(temp_663GD_per_day_tmax_tmin), len(temp_663GD_per_day_tmax_tmin.resample('D').asfreq()))  # 1826, 1826

# so 663GE, 663GB, 663 and 663GA are not continuous

# some stations do not have continuous date, we solve this problem here
temp_663GC_per_day_tmax_tmin = temp_663GC_per_day_tmax_tmin.resample('D').asfreq()
temp_663GE_per_day_tmax_tmin = temp_663GE_per_day_tmax_tmin.resample('D').asfreq()
temp_663GB_per_day_tmax_tmin = temp_663GB_per_day_tmax_tmin.resample('D').asfreq()
temp_663_per_day_tmax_tmin = temp_663_per_day_tmax_tmin.resample('D').asfreq()
temp_663GA_per_day_tmax_tmin = temp_663GA_per_day_tmax_tmin.resample('D').asfreq()
temp_663A4_per_day_tmax_tmin = temp_663A4_per_day_tmax_tmin.resample('D').asfreq()
temp_663GD_per_day_tmax_tmin = temp_663GD_per_day_tmax_tmin.resample('D').asfreq()

temp_663GC_per_day_tmax_tmin.ffill(inplace=True)
temp_663GE_per_day_tmax_tmin.ffill(inplace=True)
temp_663GB_per_day_tmax_tmin.ffill(inplace=True)
temp_663_per_day_tmax_tmin.ffill(inplace=True)
temp_663GA_per_day_tmax_tmin.ffill(inplace=True)
temp_663A4_per_day_tmax_tmin.ffill(inplace=True)
temp_663GD_per_day_tmax_tmin.ffill(inplace=True)




# climate_map = climate_df[['NAME', 'LATITUDE', 'LONGITUDE']].sort_values(['LATITUDE', 'LONGITUDE']).drop_duplicates() # map (station, x,y) to google map
# climate_map.to_clipboard()

#%% combine (precipitation, snow, max_temp, min_temp) for each clinic
prcp_snow_tmax_tmin_663GC = pd.merge(left=climate_663GC_per_day_prcp_snow, right=temp_663GC_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663GE = pd.merge(left=climate_663GE_per_day_prcp_snow, right=temp_663GE_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663GB = pd.merge(left=climate_663GB_per_day_prcp_snow, right=temp_663GB_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663 = pd.merge(left=climate_663_per_day_prcp_snow, right=temp_663_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663GA = pd.merge(left=climate_663GA_per_day_prcp_snow, right=temp_663GA_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663A4 = pd.merge(left=climate_663A4_per_day_prcp_snow, right=temp_663A4_per_day_tmax_tmin, left_index=True, right_index=True)
prcp_snow_tmax_tmin_663GD = pd.merge(left=climate_663GD_per_day_prcp_snow, right=temp_663GD_per_day_tmax_tmin, left_index=True, right_index=True)

#%% combine google and weather features
google_prcp_snow_tmax_tmin_663GC = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663GC, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663GE = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663GE, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663GB = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663GB, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663 = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663GA = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663GA, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663A4 = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663A4, left_index=True, right_index=True)
google_prcp_snow_tmax_tmin_663GD = pd.merge(left=df_google, right=prcp_snow_tmax_tmin_663GD, left_index=True, right_index=True)

# print(len(google_prcp_snow_tmax_tmin_663GC.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663GC)) # 1739, 1739
# print(len(google_prcp_snow_tmax_tmin_663GE.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663GE)) # 1738, 1738
# print(len(google_prcp_snow_tmax_tmin_663GB.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663GB)) # 1739, 1739
# print(len(google_prcp_snow_tmax_tmin_663.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663)) # 1739, 1739
# print(len(google_prcp_snow_tmax_tmin_663GA.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663GA)) # 1738, 1738
# print(len(google_prcp_snow_tmax_tmin_663A4.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663A4)) # 1739, 1739
# print(len(google_prcp_snow_tmax_tmin_663GD.resample('D').asfreq()), len(google_prcp_snow_tmax_tmin_663GD)) # 1739, 1739

#%% combine (google, weather) with date features
date_663_index = date_663.set_index('date')
date_663A4_index = date_663A4.set_index('date')
date_663GA_index = date_663GA.set_index('date')
date_663GB_index = date_663GB.set_index('date')
date_663GC_index = date_663GC.set_index('date')
date_663GD_index = date_663GD.set_index('date')
date_663GE_index = date_663GE.set_index('date')
date_663GF_index = date_663GF.set_index('date')

# check continuity (success)
# print(len(date_663_index.resample('D').asfreq()), len(date_663_index))  # 1455, 1455
# print(len(date_663A4_index.resample('D').asfreq()), len(date_663A4_index))  # 1457, 1457
# print(len(date_663GA_index.resample('D').asfreq()), len(date_663GA_index))  # 1457, 1457
# print(len(date_663GB_index.resample('D').asfreq()), len(date_663GB_index))  # 1457, 1457
# print(len(date_663GC_index.resample('D').asfreq()), len(date_663GC_index))  # 1452, 1452
# print(len(date_663GD_index.resample('D').asfreq()), len(date_663GD_index))  # 1457, 1457
# print(len(date_663GE_index.resample('D').asfreq()), len(date_663GE_index))  # 450, 450
# print(len(date_663GF_index.resample('D').asfreq()), len(date_663GF_index))  # 0, 0

external_feature_663GC= pd.merge(left=date_663GC_index, right=google_prcp_snow_tmax_tmin_663GC, left_index=True, right_index=True)
external_feature_663GE= pd.merge(left=date_663GE_index, right=google_prcp_snow_tmax_tmin_663GE, left_index=True, right_index=True)
external_feature_663GB= pd.merge(left=date_663GB_index, right=google_prcp_snow_tmax_tmin_663GB, left_index=True, right_index=True)
external_feature_663= pd.merge(left=date_663_index, right=google_prcp_snow_tmax_tmin_663, left_index=True, right_index=True)
external_feature_663GA= pd.merge(left=date_663GA_index, right=google_prcp_snow_tmax_tmin_663GA, left_index=True, right_index=True)
external_feature_663A4= pd.merge(left=date_663A4_index, right=google_prcp_snow_tmax_tmin_663A4, left_index=True, right_index=True)
external_feature_663GD= pd.merge(left=date_663GD_index, right=google_prcp_snow_tmax_tmin_663GD, left_index=True, right_index=True)

#%% reset index and save
external_feature_663GC.reset_index(drop=True).to_csv('../with_external_feature_663GC.csv', index=False)
external_feature_663GE.reset_index(drop=True).to_csv('../with_external_feature_663GE.csv', index=False)
external_feature_663GB.reset_index(drop=True).to_csv('../with_external_feature_663GB.csv', index=False)
external_feature_663.reset_index(drop=True).to_csv('../with_external_feature_663.csv', index=False)
external_feature_663GA.reset_index(drop=True).to_csv('../with_external_feature_663GA.csv', index=False)
external_feature_663A4.reset_index(drop=True).to_csv('../with_external_feature_663A4.csv', index=False)
external_feature_663GD.reset_index(drop=True).to_csv('../with_external_feature_663GD.csv', index=False)
