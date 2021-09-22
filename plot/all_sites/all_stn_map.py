"""
Map of site locations
use 'basemap' conda environment for this one
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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

clean_dir = '/data/awn/impute/awn-impute/data/clean/'
df_t = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_t = df_t.loc[~df_t.index.duplicated(keep='first')]
drop_thres = len(df_t)*0.3
na_count = df_t.isna().sum()
drop_sites = na_count[na_count > drop_thres]
#df_c_all = df_c_all.drop(columns=drop_sites.index)
df_t = df_t.drop(columns=drop_sites.index)
df_t = df_t.drop(columns=['330101'])
na_pct = df_t.isna().sum()/298753.*100


fig = plt.figure(figsize=(8, 4.5))
m = Basemap(projection='lcc', resolution='h',
            width=1.5E6*0.40, height=1E6*0.40,
            lat_0=47.30, lon_0=-120.78,)
#m.etopo(scale=0.5, alpha=0.5)
m.drawstates(linewidth=1.2)
m.drawcountries(linewidth=1.2)
m.drawcoastlines(linewidth=0.5)
m.drawcounties(linewidth=0.2)
m.drawparallels(np.linspace(-90, 90, 181), labels=[True, False, False, True])
m.drawmeridians(np.linspace(-180, 180, 181), labels=[True, False, False, True])

# Map (long, lat) to (x, y) for plotting
for i, stnid in enumerate(df.index):
    lat = df['lat'].loc[stnid]
    lon = df['lon'].loc[stnid]
    if lat > 0:
        pct = na_pct[stnid]
        x, y = m(lon, lat)
        m.scatter([x], [y], c=[pct], marker='o', cmap='viridis', vmin=0, vmax=30, s=20, edgecolor='black', linewidth=0.1)

        #plt.text(x, y, ' Seattle', fontsize=12);
cbar = plt.colorbar(extend='both')
cbar.set_label('Missing data (%)')
#plt.title('Abs Bias')
plt.savefig('FIG_MAP.png', dpi=300, bbox_inches='tight')


pdb.set_trace()
#------ bias



fig = plt.figure(figsize=(8, 4.5))
m = Basemap(projection='lcc', resolution='h',
            width=1.5E6*0.55, height=1E6*0.55,
            lat_0=47.05, lon_0=-120.9,)
#m.etopo(scale=0.5, alpha=0.5)
m.drawstates(linewidth=0.5)
m.drawcountries(linewidth=0.2)
m.drawcoastlines(linewidth=0.5)
m.drawcounties(linewidth=0.2)

# Map (long, lat) to (x, y) for plotting
for i, stnid in enumerate(df.index):
    lat = df['lat'].loc[stnid]
    lon = df['lon'].loc[stnid]
    if lat > 0:
        bias = stats['mean'][int(stnid)]
        x, y = m(lon, lat)
        m.scatter([x], [y], c=[bias], marker='o', cmap='viridis', vmin=0.6, vmax=1.05, s=20, edgecolor='black', linewidth=0.1)

        #plt.text(x, y, ' Seattle', fontsize=12);
cbar = plt.colorbar(extend='both')
cbar.set_label('Avg absolute bias (deg F)')
#plt.title('Abs Bias')
plt.savefig('bias.png', dpi=200, bbox_inches='tight')