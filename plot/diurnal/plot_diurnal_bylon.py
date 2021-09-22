"""
diurnal plots by longitude

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

seasonal_west = pd.DataFrame(index=np.linspace(1,12,12), columns=['mean', 'pct95', 'pct999'])
diurnal_west = pd.DataFrame(index=np.linspace(0,23,24), columns=['mean', 'pct95', 'pct999'])

seasonal_central = pd.DataFrame(index=np.linspace(1,12,12), columns=['mean', 'pct95', 'pct999'])
diurnal_central = pd.DataFrame(index=np.linspace(0,23,24), columns=['mean', 'pct95', 'pct999'])

seasonal_east = pd.DataFrame(index=np.linspace(1,12,12), columns=['mean', 'pct95', 'pct999'])
diurnal_east = pd.DataFrame(index=np.linspace(0,23,24), columns=['mean', 'pct95', 'pct999'])

pred = pred[stn70.index.values]
act = act[stn70.index.values]
diff = abs(pred - act)

# divide into three sections
df_west = df.index[df['lon'] < -122.0]
df_central = df.index[((df['lon'] > -122.0) & (df['lon'] < -120.0))]
df_east = df.index[df['lon'] > -120.0]

west = diff[df_west]
central = diff[df_central]
east = diff[df_east]

for i in range(1,13):
    seasonal_west['mean'][i] = np.mean(np.mean(west.loc[west.index.month == i]))
    seasonal_west['pct95'][i] = np.mean(np.nanpercentile(west.loc[west.index.month == i], 95))
    seasonal_west['pct999'][i] = np.mean(np.nanpercentile(west.loc[west.index.month == i], 99.9))

    seasonal_central['mean'][i] = np.mean(np.mean(central.loc[central.index.month == i]))
    seasonal_central['pct95'][i] = np.mean(np.nanpercentile(central.loc[central.index.month == i], 95))
    seasonal_central['pct999'][i] = np.mean(np.nanpercentile(central.loc[central.index.month == i], 99.9))

    seasonal_east['mean'][i] = np.mean(np.mean(east.loc[east.index.month == i]))
    seasonal_east['pct95'][i] = np.mean(np.nanpercentile(east.loc[east.index.month == i], 95))
    seasonal_east['pct999'][i] = np.mean(np.nanpercentile(east.loc[east.index.month == i], 99.9))

west.index = west.index + timedelta(hours=8) # UTC conversion
central.index = central.index + timedelta(hours=8) # UTC conversion
east.index = east.index + timedelta(hours=8) # UTC conversion
for i in range(0,24):
    diurnal_west['mean'][i] = np.mean(np.mean(west.loc[west.index.hour == i]))
    diurnal_west['pct95'][i] = np.mean(np.nanpercentile(west.loc[west.index.hour == i], 95))
    diurnal_west['pct999'][i] = np.mean(np.nanpercentile(west.loc[west.index.hour == i], 99.9))

    diurnal_central['mean'][i] = np.mean(np.mean(central.loc[central.index.hour == i]))
    diurnal_central['pct95'][i] = np.mean(np.nanpercentile(central.loc[central.index.hour == i], 95))
    diurnal_central['pct999'][i] = np.mean(np.nanpercentile(central.loc[central.index.hour == i], 99.9))

    diurnal_east['mean'][i] = np.mean(np.mean(east.loc[east.index.hour == i]))
    diurnal_east['pct95'][i] = np.mean(np.nanpercentile(east.loc[east.index.hour == i], 95))
    diurnal_east['pct999'][i] = np.mean(np.nanpercentile(east.loc[east.index.hour == i], 99.9))


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]

seasonal_west = seasonal_west * 5/9.
diurnal_west = diurnal_west * 5/9.

seasonal_central = seasonal_central * 5/9.
diurnal_central = diurnal_central * 5/9.

seasonal_east = seasonal_east * 5/9.
diurnal_east = diurnal_east * 5/9.


meshplot = ax1.plot(seasonal_west.index, seasonal_west['mean'],  linestyle='-', color='black', label='West')
#eshplot2 = ax1.plot(seasonal_west.index, seasonal_west['pct95'],  linestyle='-', color='blue', label='95th PCT - West')
#meshplot3 = ax1.plot(seasonal_west.index, seasonal_west['pct999'],  linestyle='-', color='purple', label='99.9th PCT - West')

meshplot = ax1.plot(seasonal_central.index, seasonal_central['mean'],  linestyle='--', color='black', label='Central')
#meshplot2 = ax1.plot(seasonal_central.index, seasonal_central['pct95'],  linestyle='--', color='blue', label='95th PCT - Central')
#meshplot3 = ax1.plot(seasonal_central.index, seasonal_central['pct999'],  linestyle='--', color='purple', label='99.9th PCT -Central')

meshplot = ax1.plot(seasonal_east.index, seasonal_east['mean'],  linestyle='dotted', color='black', label='East')
#meshplot2 = ax1.plot(seasonal_east.index, seasonal_east['pct95'],  linestyle='-.', color='blue', label='95th PCT - East')
#meshplot3 = ax1.plot(seasonal_east.index, seasonal_east['pct999'],  linestyle='-.', color='purple', label='99.9th PCT - East')

ax1.legend(loc='lower center')
ax1.set_ylabel('Mean abs error ($^{\circ}$C)')
#ax1.set_xlabel('Month')

for ax in axes:
    ax.set_xticks(np.linspace(1,12,12))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(0.5,12.5)
    ax.set_ylim(0.2,0.6)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('seasonal_3region_utc.png', dpi=300, bbox_inches='tight')
plt.close()


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]


meshplot = ax1.plot(diurnal_west.index, diurnal_west['mean'],  linestyle='-', color='black', label='West')
#meshplot2 = ax1.plot(diurnal_west.index, diurnal_west['pct95'],  linestyle='-', color='blue', label='95th PCT')
#meshplot3 = ax1.plot(diurnal_west.index, diurnal_west['pct999'],  linestyle='-', color='purple', label='99.9th PCT')

meshplot = ax1.plot(diurnal_central.index, diurnal_central['mean'],  linestyle='--', color='black', label='Central')
#meshplot2 = ax1.plot(diurnal_central.index, diurnal_central['pct95'],  linestyle='--', color='blue', label='95th PCT')
#meshplot3 = ax1.plot(diurnal_central.index, diurnal_central['pct999'],  linestyle='--', color='purple', label='99.9th PCT')

meshplot = ax1.plot(diurnal_east.index, diurnal_east['mean'],  linestyle='dotted', color='black', label='East')
#meshplot2 = ax1.plot(diurnal_east.index, diurnal_east['pct95'],  linestyle='dotted', color='blue', label='95th PCT')
#meshplot3 = ax1.plot(diurnal_east.index, diurnal_east['pct999'],  linestyle='dotted', color='purple', label='99.9th PCT')

ax1.legend()
ax1.set_ylabel('Mean abs. error ($^{\circ}$C)')
ax1.set_xlabel('Hour (UTC)')

for ax in axes:
    #ax.set_xticks(np.linspace(0,23,24))
    #ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(0,24)
    ax.set_ylim(0.2,0.6)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('diurnal_3region_utc.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()