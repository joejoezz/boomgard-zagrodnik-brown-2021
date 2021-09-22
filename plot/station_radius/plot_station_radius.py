"""
Summary plot of station radius vs. average error and 95th PCT error
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


index = [5,10,15,20,25,30,40,50,75,100,125,150,200,250,300,500,1000]
df_out = pd.DataFrame(index=index, columns=['mean', 'mean_std', 'pct95', 'pct95_std', 'time', 'n_stations', 'n_stations_std', 'n_samples', 'avg_radius'])
n_stations_scatter = []
mean_error_scatter = []
radius_scatter = []
nstations_scatter = []
radius_scatter2 = []

n_stations_scatter_w = []
mean_error_scatter_w = []
radius_scatter_w = []
nradius_scatter2_w = []
nstations_scatter_w = []
radius_scatter2_w = []

n_stations_scatter_e = []
mean_error_scatter_e = []
radius_scatter_e = []
nradius_scatter2_e = []
nstations_scatter_e = []
radius_scatter2_e = []

n_stations_scatter_c = []
mean_error_scatter_c = []
radius_scatter_c = []
nradius_scatter2_c = []
nstations_scatter_c = []
radius_scatter2_c = []

stations_within_50 = []

datadir = '/data/awn/impute/paper/data/train_predict/station_radius/'
latlon = pd.read_pickle('/data/awn/impute/paper/meta_lat_lon.p')
latlon.index = latlon.index.astype(int)

for ind in index:
    df = pd.read_csv('{}t_rh_{}km_stats.csv'.format(datadir, ind), index_col=0)
    df['n_stations'][df['n_stations'] < 1] = np.nan
    means = df.mean()
    stds = df.std()
    df_out.loc[ind]['time'] = df.max()['time']/60.
    df_out.loc[ind]['mean'] = df.mean()['mean']
    df_out.loc[ind]['mean_std'] = df.std()['mean']
    df_out.loc[ind]['pct95'] = df.mean()['pct95']
    df_out.loc[ind]['pct95_std'] = df.std()['pct95']
    df_out.loc[ind]['n_stations'] = df.mean()['n_stations']
    df_out.loc[ind]['n_stations_std'] = df.std()['n_stations']
    df_out.loc[ind]['avg_radius'] = df.mean()['mean_distance']
    df_out.loc[ind]['n_samples'] = len(df['n_stations'].dropna())

    for ind2 in df.dropna().index:
        lon = latlon['lon'].loc[ind2]
        mean_error_scatter.append(df.dropna().loc[ind2]['mean'])
        n_stations_scatter.append(df.dropna().loc[ind2]['n_stations'])
        radius_scatter.append(df.dropna().loc[ind2]['mean_distance'])
        radius_scatter2.append(ind)

        if lon < -122.0:
            mean_error_scatter_w.append(df.dropna().loc[ind2]['mean'])
            n_stations_scatter_w.append(df.dropna().loc[ind2]['n_stations'])
            radius_scatter_w.append(df.dropna().loc[ind2]['mean_distance'])
            radius_scatter2_w.append(ind)
        elif lon > -120.0:
            mean_error_scatter_e.append(df.dropna().loc[ind2]['mean'])
            n_stations_scatter_e.append(df.dropna().loc[ind2]['n_stations'])
            radius_scatter_e.append(df.dropna().loc[ind2]['mean_distance'])
            radius_scatter2_e.append(ind)
        else:
            mean_error_scatter_c.append(df.dropna().loc[ind2]['mean'])
            n_stations_scatter_c.append(df.dropna().loc[ind2]['n_stations'])
            radius_scatter_c.append(df.dropna().loc[ind2]['mean_distance'])
            radius_scatter2_c.append(ind)

    skill_e = []
    skill_c = []
    skill_w = []
    skill_over = []
    n_e = []
    n_c = []
    n_w = []
    n_over = []
    if ind == 50:
        n_within_50 = df['n_stations'].fillna(0)
    if ind == 1000:
        skill_within_50 = df['mean']
        for ind2 in n_within_50.index:
            lon = latlon['lon'].loc[ind2]
            skill_over.append(skill_within_50.loc[ind2])
            n_over.append(n_within_50.loc[ind2])
            if lon < -122.0:
                skill_w.append(skill_within_50.loc[ind2])
                n_w.append(n_within_50.loc[ind2])
            elif lon > -120.0:
                skill_e.append(skill_within_50.loc[ind2])
                n_e.append(n_within_50.loc[ind2])
            else:
                skill_c.append(skill_within_50.loc[ind2])
                n_c.append(n_within_50.loc[ind2])




pdb.set_trace()

fig = plt.figure()
fig.set_size_inches(4,4)
ax1 = fig.add_subplot(1, 1, 1)
pp = ax1.scatter(n_w, np.array(skill_w) * 5 / 9,  marker='o', s=8, color='blue', label='West')
pp2 = ax1.scatter(n_c, np.array(skill_c) * 5 / 9,  marker='o', s=8, color='black', label='Central')
pp3 = ax1.scatter(n_e, np.array(skill_e) * 5 / 9,  marker='o', s=8, color='red', label='East')
ax1.set_xlabel('Training stations within 50 km radius (#)')
ax1.set_ylabel('Mean Abs Error ($^{\circ}$C)')
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_xlim(4.5,1050)
#ax1.set_ylim(0.3,4)
#formatter = LogFormatter(10, labelOnlyBase=False)
#cbar = plt.colorbar(pp, ticks=[1, 5, 10, 50, 100], format=formatter)
#cbar = plt.colorbar(pp)#, extend='both')
#cbar.set_label('Stations within radius $\it{r}$ (#)')
#ax1.set_xticks([5,10,50,100,500,1000])
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.set_yticks([0.3, 1, 2, 3, 4])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.title("Eastern WA -- updated colormap")
plt.legend()
plt.savefig('scatter_test.eps')
plt.savefig('scatter_test.png', dpi=300, bbox_inches='tight')
plt.close()
pdb.set_trace()


mean_error_scatter_e = np.array(mean_error_scatter_e) * 5 / 9.
mean_error_scatter_w = np.array(mean_error_scatter_w) * 5 / 9.
mean_error_scatter = np.array(mean_error_scatter) * 5 / 9.

# data directory
# 2d histogram of the datasets and the differences
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
pp = ax1.scatter(radius_scatter2_e, mean_error_scatter_e, c=n_stations_scatter_e, marker='o', cmap='viridis_r', s=8,  norm=colors.LogNorm(vmin=1, vmax=133))
ax1.set_xlabel('Radius $\it{r}$ of nearest stations used for training (km)')
ax1.set_ylabel('Mean Abs Error ($^{\circ}$C)')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(4.5,1050)
ax1.set_ylim(0.3,4)
formatter = LogFormatter(10, labelOnlyBase=False)
cbar = plt.colorbar(pp, ticks=[1, 5, 10, 50, 100], format=formatter)
#cbar = plt.colorbar(pp)#, extend='both')
cbar.set_label('Stations within radius $\it{r}$ (#)')
ax1.set_xticks([5,10,50,100,500,1000])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_yticks([0.3, 1, 2, 3, 4])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.title("Eastern WA -- updated colormap")
plt.savefig('number_radius_scatter_east_v6.eps')
plt.savefig('Fig_radius_panel_A_EASTERN_WA_v6.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
pp = ax1.scatter(radius_scatter2_w, mean_error_scatter_w, c=n_stations_scatter_w, marker='o', cmap='viridis_r', s=8,  norm=colors.LogNorm(vmin=1, vmax=133))
ax1.set_xlabel('Radius $\it{r}$ of nearest stations used for training (km)')
ax1.set_ylabel('Mean Abs Error ($^{\circ}$C)')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(4.5,1050)
ax1.set_ylim(0.3,4)
formatter = LogFormatter(10, labelOnlyBase=False)
cbar = plt.colorbar(pp, ticks=[1, 5, 10, 50, 100], format=formatter)
#cbar = plt.colorbar(pp)#, extend='both')
cbar.set_label('Stations within radius $\it{r}$ (#)')
ax1.set_xticks([5,10,50,100,500,1000])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_yticks([0.3, 1, 2, 3, 4])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.title("Eastern WA -- updated colormap")
plt.savefig('number_radius_scatter_west_v6.eps')
plt.savefig('Fig_radius_panel_B_WESTERN_WA_v6.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
pp = ax1.scatter(radius_scatter2_c, mean_error_scatter_c, c=n_stations_scatter_c, marker='o', cmap='viridis_r', s=8,  norm=colors.LogNorm(vmin=1, vmax=133))
ax1.set_xlabel('Radius $\it{r}$ of nearest stations used for training (km)')
ax1.set_ylabel('Mean Abs Error ($^{\circ}$C)')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(4.5,1050)
ax1.set_ylim(0.3,4)
formatter = LogFormatter(10, labelOnlyBase=False)
cbar = plt.colorbar(pp, ticks=[1, 5, 10, 50, 100], format=formatter)
#cbar = plt.colorbar(pp)#, extend='both')
cbar.set_label('Stations within radius $\it{r}$ (#)')
ax1.set_xticks([5,10,50,100,500,1000])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_yticks([0.3, 1, 2, 3, 4])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.title("Eastern WA -- updated colormap")
plt.savefig('number_radius_scatter_central_v6.eps')
plt.savefig('Fig_radius_panel_B_CENTRAL_WA_v6.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()

fig = plt.figure()
fig.set_size_inches(5,7)
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)
axes=[ax1,ax2, ax3, ax4]

meshplot = ax1.errorbar(df_out.index, df_out['mean'], yerr=df_out['mean_std'], linestyle='--', marker='o', color='k', label='Mean Error')
meshplot = ax2.errorbar(df_out.index, df_out['pct95'], yerr=df_out['pct95_std'], linestyle='--', marker='o', color='b', label='95th Pct. Error')
meshplot = ax3.plot(df_out.index, df_out['n_samples'], linestyle='--', marker='o', color='red', label='# imputing samples')
meshplot = ax4.errorbar(df_out.index, df_out['n_stations'], yerr=df_out['n_stations_std'], linestyle='--', marker='o', color='orange', label='Avg. # stations within radius')
ax1.legend()
ax2.legend()
ax3.legend(loc='lower right')
ax4.legend(loc='upper left')

ax4.set_xlabel('Radius of stations used for imputing (km)')
ax1.set_ylabel('Error ($^{\circ}$F)')
ax2.set_ylabel('Error ($^{\circ}$F)')
ax3.set_ylabel('# samples')
ax4.set_ylabel('avg. # stations')
for ax in axes:
    ax.set_xticks(index)
    ax.grid(axis='y')
    ax.set_xscale('log')
    ax.set_xlim(4.5,500)
    ax.set_xticklabels(['','','10','100'])
#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('station_radius_v3.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()