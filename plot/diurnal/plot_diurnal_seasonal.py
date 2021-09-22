"""
Map of site locations
use 'basemap' conda environment for this one
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import pdb
import glob
import geopy.distance
from datetime import timedelta


def get_distance_from_station(stid, meta):
    """
    gets the distance to all of the stations in km
    """
    distances = pd.Series(index=meta.index)
    for column in columns:
        coords_1 = (meta.loc[stid]['lat'], meta.loc[stid]['lon'])
        coords_2 = (meta.loc[str(column)]['lat'], meta.loc[str(column)]['lon'])
        distance = geopy.distance.great_circle(coords_1, coords_2).km
        distances.loc[column] = distance
        distances = distances.sort_values()

    return distances



exp = 't_rh_rf'
datadir = '/data/awn/impute/paper/data/train_predict/all_sites/'
pred = pd.read_pickle('{}{}_predict.p'.format(datadir, exp))
act = pd.read_pickle('{}{}_actual.p'.format(datadir, exp))
stats = pd.read_csv('{}{}_stats.csv'.format(datadir, exp), index_col=0)

df = pd.read_pickle('/data/awn/impute/awn-impute/paper/fig1_map/station_metadata.p')

stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
stn70 = stn70.drop('330101')
ind70 = stn70.index.astype('int')

df = df.loc[stn70.index]
df.lon['300029'] = -117.148
df.lat['300029'] = 46.69636
df.lon['100129'] = -119.769
df.lat['100129'] = 46.07730979
stats = stats.loc[ind70]
columns = stats.index
naches = get_distance_from_station('310126', df)



seasonal = pd.DataFrame(index=np.linspace(1,12,12), columns=['mean', 'pct95', 'pct999'])
diurnal = pd.DataFrame(index=np.linspace(0,23,24), columns=['mean', 'pct95', 'pct999'])

pred = pred[stn70.index.values]
act = act[stn70.index.values]
diff = abs(pred - act)

for i in range(1,13):
    seasonal['mean'][i] = np.mean(np.mean(diff.loc[diff.index.month == i]))
    seasonal['pct95'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.month == i], 95))
    seasonal['pct999'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.month == i], 99.9))

diff.index = diff.index + timedelta(hours=8) # UTC conversion
for i in range(0,24):
    diurnal['mean'][i] = np.mean(np.mean(diff.loc[diff.index.hour == i]))
    diurnal['pct95'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.hour == i], 95))
    diurnal['pct999'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.hour == i], 99.9))


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]

seasonal = seasonal * 5/9.
diurnal = diurnal * 5/9.


meshplot = ax1.plot(seasonal.index, seasonal['mean'],  linestyle='-', color='black', label='Mean')
meshplot2 = ax1.plot(seasonal.index, seasonal['pct95'],  linestyle='-', color='blue', label='95th PCT')
meshplot3 = ax1.plot(seasonal.index, seasonal['pct999'],  linestyle='-', color='purple', label='99.9th PCT')
ax1.legend()
ax1.set_ylabel('Mean error ($^{\circ}$C)')
#ax1.set_xlabel('Month')

for ax in axes:
    ax.set_xticks(np.linspace(1,12,12))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(0.5,12.5)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('seasonal.png', dpi=300, bbox_inches='tight')
plt.close()


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]


meshplot = ax1.plot(diurnal.index, diurnal['mean'],  linestyle='-', color='black', label='Mean')
meshplot2 = ax1.plot(diurnal.index, diurnal['pct95'],  linestyle='-', color='blue', label='95th PCT')
meshplot3 = ax1.plot(diurnal.index, diurnal['pct999'],  linestyle='-', color='purple', label='99.9th PCT')
ax1.legend()
ax1.set_ylabel('Mean error ($^{\circ}$C)')
ax1.set_xlabel('Hour (UTC)')

for ax in axes:
    #ax.set_xticks(np.linspace(0,23,24))
    #ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(-0.5,23.5)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('diurnal_uct.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()