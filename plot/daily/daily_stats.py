"""
stats for all stations
"""


import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


exp_names = ['t_all_include_stn_rf', 't_rh_include_stn_rf', 't_solar_include_stn_rf', 't_wspd_include_stn_rf']
df_out = pd.DataFrame(index=exp_names, columns=['mean', 'pct95', 'pct99', 'pct999'])
datadir = '/data/awn/impute/paper/data/train_predict/all_sites_include_stn/'
stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
ind70 = stn70.index.astype('int')
ind70 = ind70.drop(330101)

"""

for exp in exp_names:
    df = pd.read_csv('{}{}_stats.csv'.format(datadir, exp), index_col=0)
    pdb.set_trace()
    df_out.loc[exp]['mean'] = df.mean()['mean']
    df_out.loc[exp]['pct95'] = df.mean()['pct95']
    df_out.loc[exp]['pct99'] = df.mean()['pct99']
    df_out.loc[exp]['pct999'] = df.mean()['pct999']

pdb.set_trace()


"""

exp_names = ['t_all_lin', 't_all_rf', 't_only_lin', 't_only_rf', 't_rh_lin', 't_rh_rf', 't_solar_lin', 't_solar_rf', 't_u_v_lin', 't_u_v_rf', 't_wdir_lin', 't_wdir_rf', 't_wspd_rf', 't_wspd_lin']
exp_names = ['nn_15min', 't_only_lin', 't_only_rf', 't_rh_rf', 't_solar_rf', 't_wspd_rf', 't_wdir_rf', 't_all"rf']

df_out = pd.DataFrame(index=exp_names, columns=['mean', 'pct95', 'pct99', 'pct999', 'nstn'])
datadir = '/data/awn/impute/paper/data/train_predict/all_sites/'

for exp in exp_names:
    df = pd.read_csv('{}{}_stats.csv'.format(datadir, exp), index_col=0)
    df = df.loc[ind70]
    df_out.loc[exp]['nstn'] = len(df.dropna())
    df_out.loc[exp]['mean'] = np.round(df.mean()['mean'] * 5/9.,2)
    df_out.loc[exp]['pct95'] = np.round(df.mean()['pct95'] * 5/9.,2)
    df_out.loc[exp]['pct99'] = np.round(df.mean()['pct99'] * 5/9.,2)
    df_out.loc[exp]['pct999'] = np.round(df.mean()['pct999'] * 5/9.,2)


pdb.set_trace()