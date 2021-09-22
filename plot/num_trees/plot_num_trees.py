"""
Summary plot of station radius vs. average error and 95th PCT error
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

index_tree = [1,1,1,5,5,5,10,10,10,15,15,15,20,20,20,25,25,25,30,30,30,50,50,50,100,100,100]
index_stn = [1,20,100]*9
arrays = [index_tree,index_stn]
tuples = list(zip(*arrays))
multi = pd.MultiIndex.from_tuples(tuples, names=['tree', 'stn'])

df_out = pd.DataFrame(index=multi, columns=['mean', 'mean_std', 'pct95', 'pct95_std', 'time'])
df_out['stn']  = index_stn

datadir = '/data/awn/impute/paper/data/train_predict/num_trees/'
latlon = pd.read_pickle('/data/awn/impute/paper/meta_lat_lon.p')
latlon.index = latlon.index.astype(int)

for ind in index_tree:
    for ind2 in index_stn:
        df = pd.read_csv('{}t_rh_{}stn_{}tree_stats.csv'.format(datadir, ind2, ind), index_col=0)
        df_out['time'][(ind,ind2)] = df.max()['time']/60.
        df_out['mean'][(ind, ind2)] = df.mean()['mean']
        df_out['mean_std'][(ind, ind2)] = df.std()['mean']
        df_out['pct95'][(ind, ind2)] = df.mean()['pct95']
        df_out['pct95_std'][(ind, ind2)] = df.std()['pct95']

tree_100 = df_out.xs(100,level='stn')
fig = plt.figure()
fig.set_size_inches(5,3)
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
meshplot = ax1.plot(tree_100.index, tree_100['mean']*5/9.,  linestyle='--', marker='o', color='blue', lw=1, ms=4)
meshplot = ax2.plot(tree_100.index, tree_100['time']/60.,  linestyle='--', marker='o', color='red', lw=1, ms=4)

ax1.set_ylabel('Mean abs. error ($^{\circ}$C)', color='blue')
ax2.set_ylabel('Model training time (hr)', color='red')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax1.grid()

ax1.set_xticks(tree_100.index)
#ax.grid(axis='y')
ax1.set_xscale('log')
ax1.set_xlim(0.8, 110)
ax2.set_ylim(0,125)
ax1.set_xticklabels(['', '', '1', '10', '100'])
ax1.set_xlabel('Number of trees')

plt.savefig('Fig_HYPER.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig_HYPER.eps', bbox_inches='tight')
plt.close()
pdb.set_trace()

fig = plt.figure()
fig.set_size_inches(5,7)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
axes=[ax1,ax2, ax3]

tree_1 = df_out.xs(1,level='stn')
tree_20 = df_out.xs(20,level='stn')
tree_100 = df_out.xs(100,level='stn')
meshplot = ax1.plot(tree_1.index, tree_1['mean'],  linestyle='--', marker='o', color='red', label='1 stn')
meshplot = ax1.plot(tree_20.index, tree_20['mean'],  linestyle='--', marker='o', color='blue', label='20 stns')
meshplot = ax1.plot(tree_100.index, tree_100['mean'],  linestyle='--', marker='o', color='purple', label='100 stns')

meshplot = ax2.plot(tree_1.index, tree_1['pct95'],  linestyle='--', marker='o', color='red', label='1 stn')
meshplot = ax2.plot(tree_20.index, tree_20['pct95'],  linestyle='--', marker='o', color='blue', label='20 stns')
meshplot = ax2.plot(tree_100.index, tree_100['pct95'],  linestyle='--', marker='o', color='purple', label='100 stns')

meshplot = ax3.plot(tree_1.index, tree_1['time'],  linestyle='--', marker='o', color='red', label='1 stn')
meshplot = ax3.plot(tree_20.index, tree_20['time'],  linestyle='--', marker='o', color='blue', label='20 stns')
meshplot = ax3.plot(tree_100.index, tree_100['time'],  linestyle='--', marker='o', color='purple', label='100 stns')


ax1.legend()


ax1.set_ylabel('Mean error ($^{\circ}$F)')
ax2.set_ylabel('95pct error ($^{\circ}$F)')
ax3.set_ylabel('time (min)')
ax3.set_xlabel('number of trees')

for ax in axes:
    ax.set_xticks(tree_1.index)
    ax.grid(axis='y')
    ax.set_xscale('log')
    ax.set_xlim(0.8,110)
    ax.set_xticklabels(['','','1','10','100'])
#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('num_trees.png', dpi=300, bbox_inches='tight')
plt.close()

pdb.set_trace()