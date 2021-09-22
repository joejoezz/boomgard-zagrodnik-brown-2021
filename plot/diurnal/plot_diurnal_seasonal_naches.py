"""
Map of site locations
use 'basemap' conda environment for this one
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import timedelta
#import scipy.stats as st
import pdb
import glob


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
stats = stats.loc[ind70]

seasonal = pd.DataFrame(index=np.linspace(1,12,12), columns=['mean', 'pct95', 'pct999'])
diurnal = pd.DataFrame(index=np.linspace(0,23,24), columns=['mean', 'pct95', 'pct999'])

pred = pred[stn70.index.values]
act = act[stn70.index.values]
diff = abs(pred - act)
diff_raw = pred - act

# naches
diff = diff['310126']
diff_raw = diff_raw['310126']
diff_raw.index = diff_raw.index + timedelta(hours=8) # UTC conversion

for i in range(1,13):
    seasonal['mean'][i] = np.mean(np.mean(diff.loc[diff.index.month == i]))
    seasonal['pct95'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.month == i], 95))
    seasonal['pct999'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.month == i], 99.9))

for i in range(0,24):
    diurnal['mean'][i] = np.mean(np.mean(diff.loc[diff.index.hour == i]))
    diurnal['pct95'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.hour == i], 95))
    diurnal['pct999'][i] = np.mean(np.nanpercentile(diff.loc[diff.index.hour == i], 99.9))

diff_gt_5 = diff_raw[np.where((diff_raw > 5))[0]]
diff_lt_5 = diff_raw[np.where((diff_raw < -5))[0]]

index_gt_5 = []
index_lt_5 = []
month = []

for ind in diff_gt_5.index:
    hour = ind.hour
    min = ind.minute / 15 / 4
    index_gt_5.append(hour+min)
    month.append(ind.month)

for ind in diff_lt_5.index:
    hour = ind.hour
    min = ind.minute / 15 / 4
    index_lt_5.append(hour+min)
    month.append(ind.month)

month = np.array(month)
winter = len(np.where((month == 12) | (month < 3))[0])
spring = len(np.where((month > 3) & (month < 6))[0])
summer = len(np.where((month > 5) & (month < 9))[0])
fall = len(np.where((month > 8) & (month < 12))[0])


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]

meshplot = ax1.scatter(index_gt_5, diff_gt_5*5/9.,  marker='o', s=2, color='purple', label='Naches')
meshplot2 = ax1.scatter(index_lt_5, diff_lt_5*5/9.,  marker='o', s=2, color='purple')
#ax1.legend()
ax1.set_ylabel('Predict - Obs ($^{\circ}$C)')
ax1.set_xlabel('Hour (UTC)')

for ax in axes:
    #ax.set_xticks(np.linspace(0,23,24))
    #ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(-0.5,24.5)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('naches_diurnal_outliers.png', dpi=300, bbox_inches='tight')
plt.close()


pdb.set_trace()


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]

meshplot = ax1.plot(seasonal.index, seasonal['mean'],  linestyle='-', color='black', label='Mean')
meshplot2 = ax1.plot(seasonal.index, seasonal['pct95'],  linestyle='-', color='blue', label='95th PCT')
meshplot3 = ax1.plot(seasonal.index, seasonal['pct999'],  linestyle='-', color='purple', label='99.9th PCT')
ax1.legend()
ax1.set_ylabel('Mean error ($^{\circ}$F)')
#ax1.set_xlabel('Month')

for ax in axes:
    ax.set_xticks(np.linspace(1,12,12))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(0.5,12.5)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('naches_seasonal.png', dpi=300, bbox_inches='tight')
plt.close()


fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
axes=[ax1]

meshplot = ax1.plot(diurnal.index, diurnal['mean'],  linestyle='-', color='black', label='Mean')
meshplot2 = ax1.plot(diurnal.index, diurnal['pct95'],  linestyle='-', color='blue', label='95th PCT')
meshplot3 = ax1.plot(diurnal.index, diurnal['pct999'],  linestyle='-', color='purple', label='99.9th PCT')
ax1.legend()
ax1.set_ylabel('Mean error ($^{\circ}$F)')
ax1.set_xlabel('Hour (local time)')

for ax in axes:
    #ax.set_xticks(np.linspace(0,23,24))
    #ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #ax.grid(axis='y')
    #ax.set_xscale('log')
    ax.set_xlim(-0.5,23.5)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('naches_diurnal.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()