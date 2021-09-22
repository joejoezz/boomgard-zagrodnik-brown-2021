"""
correlation matrix
--first sort the data by longitude
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


def get_distance_from_station(stid, meta):
    """
    gets the distance to all of the stations in km
    """
    distances = pd.Series(index=columns)
    for column in columns:
        coords_1 = (meta.loc[stid]['lat'], meta.loc[stid]['lon'])
        coords_2 = (meta.loc[column]['lat'], meta.loc[column]['lon'])
        distance = geopy.distance.great_circle(coords_1, coords_2).km
        distances.loc[column] = distance
        distances = distances.sort_values()

    return distances

# read in raw T0, T1
save_dir = '/data/awn/impute/awn-impute/data/'
df_t0 = pd.read_pickle('{}full_t_raw.p'.format(save_dir))
df_t1 = pd.read_pickle('{}full_t1_raw.p'.format(save_dir))
df_t0 = df_t0.loc[~df_t0.index.duplicated(keep='first')]
df_t1 = df_t1.loc[~df_t1.index.duplicated(keep='first')]
mean_diff = (df_t0-df_t1).mean()


df = pd.read_pickle('/data/awn/impute/awn-impute/paper/fig1_map/station_metadata.p')

stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
stn70 = stn70.drop('330101')
ind70 = stn70.index.astype('int')

df = df.loc[stn70.index]
df.lon['300029'] = -117.148
df.lat['300029'] = 46.69636
df.lon['100129'] = -119.769
df.lat['100129'] = 46.07730979

df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_temp.index = pd.to_datetime(df_temp.index)
df_temp = df_temp[stn70.index]
df['lon'] = np.round(df['lon'],2)
df_temp.columns = df['lon']
df_temp = df_temp.sort_index(axis=1)
df_temp.columns = np.round(np.sort(df['lon']),1)

cc = df_temp.corr()
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask, cmap='magma')
ax.set_xlabel('T (station longitude)')
ax.set_ylabel('T (station longitude)')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
plt.savefig('panel_a_temp.png', dpi=300, bbox_inches='tight')
plt.savefig('panel_a_temp.eps', bbox_inches='tight')
print(np.nanmin(cc))


df_rh = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/rh_clean.p')
df_rh.index = pd.to_datetime(df_rh.index)
df_rh = df_rh[stn70.index]
df_rh.columns = df['lon']
df_rh = df_rh.sort_index(axis=1)
df_rh.columns = np.round(np.sort(df['lon']),1)
#df_rh.columns = np.linspace(0,133,134)

# don't need this one
cc = df_rh.corr()
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask)#, vmax=1.0, vmin=0.8)
ax.set_xlabel('RH (station longitude)')
ax.set_ylabel('RH (station longitude)')
plt.savefig('rh.png', dpi=300, bbox_inches='tight')



concat = pd.concat((df_temp, df_rh), axis=1)
cc = concat.corr()
cc = cc.iloc[134:]
cc = cc.iloc[:,0:134]
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask, cmap='magma_r', vmax=-0.2, vmin=-0.8)
ax.set_xlabel('T (station longitude)')
ax.set_ylabel('RH (station longitude)')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
plt.savefig('panel_b_temp_rh.png', dpi=300, bbox_inches='tight')
plt.savefig('panel_b_temp_rh.eps', bbox_inches='tight')
print('rh')
print(np.nanmin(cc))

df_solar = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/solar_clean.p')
df_solar.index = pd.to_datetime(df_solar.index)
df_solar = df_solar[stn70.index]
df_solar.columns = df['lon']
df_solar = df_solar.sort_index(axis=1)
df_solar.columns = np.round(np.sort(df['lon']),1)
cc = df_solar.corr()
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask)#, vmax=1.0, vmin=0.8)
ax.set_xlabel('Solar')
ax.set_ylabel('Solar')
plt.savefig('solar.png', dpi=300, bbox_inches='tight')
concat = pd.concat((df_temp, df_solar), axis=1)
cc = concat.corr()
cc = cc.iloc[134:]
cc = cc.iloc[:,0:134]
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask, cmap='magma')#, vmax=-0.2, vmin=-0.8)
ax.set_xlabel('T (station longitude)')
ax.set_ylabel('SOL (station longitude)')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
plt.savefig('panel_c_temp_solar.png', dpi=300, bbox_inches='tight')
plt.savefig('panel_c_temp_solar.eps', bbox_inches='tight')
print('sol')
print(np.nanmin(cc))
pdb.set_trace()

df_wdir = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/dir_clean.p')
df_wdir.index = pd.to_datetime(df_wdir.index)
df_wdir = df_wdir[stn70.index]
df_wdir.columns = df['lon']
df_wdir = df_wdir.sort_index(axis=1)
df_wdir.columns = np.round(np.sort(df['lon']),1)
cc = df_wdir.corr()

mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask)#, vmax=1.0, vmin=0.8)
ax.set_xlabel('wdir')
ax.set_ylabel('wdir')
plt.savefig('wdir.png', dpi=300, bbox_inches='tight')
concat = pd.concat((df_temp, df_wdir), axis=1)
cc = concat.corr()
cc = cc.iloc[134:]
cc = cc.iloc[:,0:134]
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask, cmap='magma')#, vmax=-0.2, vmin=-0.8)
ax.set_xlabel('T (station longitude)')
ax.set_ylabel('WDIR (station longitude)')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
plt.savefig('panel_e_temp_wdir.png', dpi=300, bbox_inches='tight')
plt.savefig('panel_e_temp_wdir.eps', bbox_inches='tight')
print('wdir')
print(np.nanmin(cc))


df_wspd = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/speed_clean.p')
df_wspd.index = pd.to_datetime(df_wspd.index)
df_wspd = df_wspd[stn70.index]
df_wspd.columns = df['lon']
df_wspd = df_wspd.sort_index(axis=1)
df_wspd.columns = np.round(np.sort(df['lon']),1)
cc = df_wspd.corr()
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask)#, vmax=1.0, vmin=0.8)
ax.set_xlabel('wspd')
ax.set_ylabel('wspd')
plt.savefig('wspd.png', dpi=300, bbox_inches='tight')
concat = pd.concat((df_temp, df_wspd), axis=1)
cc = concat.corr()
cc = cc.iloc[134:]
cc = cc.iloc[:,0:134]
mask = np.zeros_like(cc)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(5,5))
ax = sns.heatmap(cc, mask=mask, cmap='magma')#, vmax=-0.2, vmin=-0.8)
ax.set_xlabel('T (station longitude)')
ax.set_ylabel('WSPD (station longitude)')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
print('wspd')
print(np.nanmin(cc))
plt.savefig('panel_d_temp_wspd.png', dpi=300, bbox_inches='tight')
plt.savefig('panel_d_temp_wspd.eps', bbox_inches='tight')


pdb.set_trace()

