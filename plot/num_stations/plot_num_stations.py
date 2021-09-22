"""
Summary plot of reduced training percentage vs. average error and 95th PCT error
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

index = [1,2,3,4,5,10,15,20,25,30,40,50,100,133]
df_out = pd.DataFrame(index=index, columns=['mean', 'mae_05', 'mae_95', 'pct95_05', 'pct95_95', 'mean_std', 'pct95', 'pct95_std', 'time'])
datadir = '/data/awn/impute/paper/data/train_predict/num_stations/'

scatter_x_mean = np.array([])
scatter_y_mean = np.array([])
scatter_x_pct95 = np.array([])
scatter_y_pct95 = np.array([])

for ind in index[0:-1]:
    df = pd.read_csv('{}t_rh_{}stn_stats.csv'.format(datadir, ind), index_col=0)
    df_actual = pd.read_pickle('{}t_rh_{}stn_actual.p'.format(datadir, ind))
    df_predict = pd.read_pickle('{}t_rh_{}stn_predict.p'.format(datadir, ind))
    means = df.mean()
    stds = df.std()
    df_out.loc[ind]['time'] = df.max()['time']/60.
    df_out.loc[ind]['mean'] = df.mean()['mean']
    df_out.loc[ind]['mean_std'] = df.std()['mean']
    df_out.loc[ind]['pct95'] = df.mean()['pct95']
    df_out.loc[ind]['pct95_std'] = df.std()['pct95']
    df_out.loc[ind]['pct95_05'] = np.abs(df['pct95'].mean() - df['pct95'].quantile(0.05))
    df_out.loc[ind]['pct95_95'] = np.abs(df['pct95'].mean() - df['pct95'].quantile(0.95))
    df_out.loc[ind]['mae_05'] = np.abs(df['mean'].mean() - df['mean'].quantile(0.05))
    df_out.loc[ind]['mae_95'] = np.abs(df['mean'].mean() - df['mean'].quantile(0.95))

    # pull values outside of quantiles for scatterplotting
    y_vals = df['mean'][df['mean'] < df['mean'].quantile(0.05)].values
    x_vals = np.ones(len(y_vals))*ind
    scatter_x_mean = np.hstack((scatter_x_mean, x_vals))
    scatter_y_mean = np.hstack((scatter_y_mean, y_vals))
    y_vals = df['mean'][df['mean'] > df['mean'].quantile(0.95)].values
    x_vals = np.ones(len(y_vals))*ind
    scatter_x_mean = np.hstack((scatter_x_mean, x_vals))
    scatter_y_mean = np.hstack((scatter_y_mean, y_vals))

    y_vals = df['pct95'][df['pct95'] < df['pct95'].quantile(0.05)].values
    x_vals = np.ones(len(y_vals))*ind
    scatter_x_pct95 = np.hstack((scatter_x_pct95, x_vals))
    scatter_y_pct95 = np.hstack((scatter_y_pct95, y_vals))
    y_vals = df['pct95'][df['pct95'] > df['pct95'].quantile(0.95)].values
    x_vals = np.ones(len(y_vals))*ind
    scatter_x_pct95 = np.hstack((scatter_x_pct95, x_vals))
    scatter_y_pct95 = np.hstack((scatter_y_pct95, y_vals))

# celsius conversion
df_out = df_out * (5/9.)
scatter_y_mean = scatter_y_mean * (5/9.)
scatter_y_pct95 = scatter_y_pct95 * (5/9.)
# data directory

# 2d histogram of the datasets and the differences
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(5,5)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
axes=[ax1,ax2]
#meshplot = ax1.errorbar(df_out.index-0.25, df_out['mean'], yerr=df_out['mean_std'], linestyle='--', marker='o', color='k', label='Mean Error')
#meshplot = ax2.errorbar(df_out.index+0.25, df_out['pct95'], yerr=df_out['pct95_std'], linestyle='--', marker='o', color='b', label='95th Pct. Error')
meshplot = ax1.errorbar(df_out.index, df_out['mean'], yerr=np.vstack((df_out['mae_05'].values, df_out['mae_95'].values)), linestyle='--', marker='o', color='k')
meshplot = ax2.errorbar(df_out.index, df_out['pct95'], yerr=(df_out['pct95_05'],df_out['pct95_95']), linestyle='--', marker='o', color='b')
ax1.scatter(scatter_x_mean, scatter_y_mean, c='white', marker='o', linewidths=0.4, edgecolors='k', s=8)
ax2.scatter(scatter_x_pct95, scatter_y_pct95, c='white', marker='o', linewidths=0.4, edgecolors='blue', s=8)


ax2.set_xlabel('Nearest stations used for training (#)')
ax1.set_ylabel('Mean Abs. Error ($^{\circ}$C)')
ax2.set_ylabel('95th Pct. Abs. Error ($^{\circ}$C)')
for ax in axes:
    ax.set_xticks(index)
    ax.set_xscale('log')
    ax.set_xticklabels(['','','1','10','100','',''])
    ax.grid(axis='y')
#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('fig_NSTATIONS_final.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_NSTATIONS_final.eps', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()


datadir = '/data/awn/impute/paper/data/train_predict/all_sites/'
df = pd.read_csv('{}t_rh_rf_stats.csv'.format(datadir, ind), index_col=0)
means = df.mean()
stds = df.std()
df_out.loc[133]['time'] = df.max()['time'] / 60.
df_out.loc[133]['mean'] = df.mean()['mean']
df_out.loc[133]['mean_std'] = df.std()['mean']
df_out.loc[133]['pct95'] = df.mean()['pct95']
df_out.loc[133]['pct95_std'] = df.std()['pct95']

# data directory
df_out = df_out * 5/9.

# 2d histogram of the datasets and the differences
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(5,5)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
axes=[ax1,ax2]
meshplot = ax1.errorbar(df_out.index, df_out['mean'], yerr=df_out['mean_std'], linestyle='--', marker='o', color='k', label='Mean Error')
meshplot = ax2.errorbar(df_out.index, df_out['pct95'], yerr=df_out['pct95_std'], linestyle='--', marker='o', color='b', label='95th Pct. Error')
ax1.legend()
ax2.legend()

ax2.set_xlabel('# nearest stations used for training')
ax1.set_ylabel('Abs. Error ($^{\circ}$C)')
ax2.set_ylabel('Abs. Error ($^{\circ}$C)')
for ax in axes:
    ax.set_xticks(index)
    ax.set_xscale('log')
    ax.set_xticklabels(['','','1','10','100','',''])
    ax.grid(axis='y')
#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('num_stations_degc.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()