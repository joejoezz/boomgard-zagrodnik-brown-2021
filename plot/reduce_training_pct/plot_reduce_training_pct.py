"""
Summary plot of reduced training percentage vs. average error and 95th PCT error
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

index = [1, 5,10,20,30,40,50,60,70,80,90]
df_out = pd.DataFrame(index=index, columns=['rmse', 'rmse_max', 'rmse_min', 'mean', 'mean_std', 'mae_min', 'mae_max', 'pct95', 'pct95_std', 'pct95_min', 'pct95_max', 'time', 'mae_25', 'mae_75', 'pct95_25', 'pct95_75'])
datadir = '/data/awn/impute/paper/data/train_predict/reduce_training_pct/'


# try pulling actual data to compute RMSE to add to MAE
scatter_x_mean = np.array([])
scatter_y_mean = np.array([])
scatter_x_pct95 = np.array([])
scatter_y_pct95 = np.array([])

for ind in index:
    df_actual = pd.read_pickle('{}t_rh_{}_actual.p'.format(datadir, ind))
    df_predict = pd.read_pickle('{}t_rh_{}_predict.p'.format(datadir, ind))
    df = pd.read_csv('{}t_rh_{}_stats.csv'.format(datadir, ind), index_col=0)
    rmse = ((df_actual - df_predict) ** 2).mean() ** .5
    means = df.mean()
    stds = df.std()
    df_out.loc[ind]['rmse'] = rmse.mean()
    df_out.loc[ind]['rmse_max'] = rmse.max()
    df_out.loc[ind]['rmse_min'] = rmse.min()
    df_out.loc[ind]['mae_max'] = df['mean'].max()
    df_out.loc[ind]['mae_min'] = df['mean'].min()
    df_out.loc[ind]['time'] = df.max()['time']/60.
    df_out.loc[ind]['mean'] = df.mean()['mean']
    df_out.loc[ind]['mean_std'] = df.std()['mean']
    df_out.loc[ind]['pct95'] = df.mean()['pct95']
    df_out.loc[ind]['pct95_std'] = df.std()['pct95']
    df_out.loc[ind]['pct95_max'] = df.pct95.max()
    df_out.loc[ind]['pct95_min'] = df.pct95.min()
    df_out.loc[ind]['pct95_25'] = np.abs(df['pct95'].mean() - df['pct95'].quantile(0.05))
    df_out.loc[ind]['pct95_75'] = np.abs(df['pct95'].mean() - df['pct95'].quantile(0.95))
    df_out.loc[ind]['mae_25'] = np.abs(df['mean'].mean() - df['mean'].quantile(0.05))
    df_out.loc[ind]['mae_75'] = np.abs(df['mean'].mean() - df['mean'].quantile(0.95))

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
meshplot = ax1.errorbar(df_out.index, df_out['mean'], yerr=np.vstack((df_out['mae_25'].values, df_out['mae_75'].values)), linestyle='--', marker='o', color='k')
meshplot = ax2.errorbar(df_out.index, df_out['pct95'], yerr=(df_out['pct95_25'],df_out['pct95_75']), linestyle='--', marker='o', color='b')
ax1.scatter(scatter_x_mean, scatter_y_mean, c='white', marker='o', linewidths=0.4, edgecolors='k', s=8)
ax2.scatter(scatter_x_pct95, scatter_y_pct95, c='white', marker='o', linewidths=0.4, edgecolors='blue', s=8)


ax2.set_xlabel('Data used for training (%)')
ax1.set_ylabel('Mean Abs. Error ($^{\circ}$C)')
ax2.set_ylabel('95th Pct. Abs. Error ($^{\circ}$C)')
for ax in axes:
    ax.set_xticks(index)
    ax.grid(axis='y')
#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('fig_PCT_final.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_PCT_final.eps', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()