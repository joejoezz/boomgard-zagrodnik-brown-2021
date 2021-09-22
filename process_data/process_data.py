"""
Create final imputation dataset -- this one is for other variables, not air temperature
Step 1: generate_full_dataset_all_vars.py and _raw.py
Step 2: this code
Step 3: run model (in all_sites)

For this code:
Load temperature dataset
Load other dataset
Remove where temperature is NA
Remove where raw is different than corrected (might not be any)?
Any other checks
For wind, convert to u,v
For solar, do log transform


"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.dates import YearLocator, DateFormatter

def wind_speed_dir_to_uv(vel, wdir):
    """
    Converts a wind speed and direction (degrees) to U and V components
    """
    polar_dir = 90 - wdir
    polar_dir_rad = np.radians(polar_dir)
    u = -vel * np.cos(polar_dir_rad)
    v = -vel * np.sin(polar_dir_rad)
    return u, v


temp_dir = '/data/awn/impute/awn-impute/data/'
out_dir = '/data/awn/impute/awn-impute/data/clean/'
df_temperature = pd.read_pickle('../data/impute_v1_air_t.p')
df_temperature = df_temperature.loc[~df_temperature.index.duplicated(keep='first')]

data_dir = '/data/awn/impute/awn-impute/data/jul_28/'

# load solar data, trim to 2012-present
df_sol = pd.read_csv('{}awn_solar_data.csv'.format(data_dir), index_col=0)
df_sol = df_sol.loc['2012-01-01 00:00:00':]
df_sol_raw = pd.read_csv('{}awn_solar_data_raw.csv'.format(data_dir), index_col=0)
df_sol_raw = df_sol_raw.loc['2012-01-01 00:00:00':]
df_sol_raw['330101'] = df_sol['330101']
diff_sol = df_sol - df_sol_raw

# load rh data, trim to 2012-present
df_rh = pd.read_csv('{}awn_rh_data.csv'.format(data_dir), index_col=0)
df_rh = df_rh.loc['2012-01-01 00:00:00':]
df_rh_raw = pd.read_csv('{}awn_rh_data_raw.csv'.format(data_dir), index_col=0)
df_rh_raw = df_rh_raw.loc['2012-01-01 00:00:00':]
# fix mt vernon
df_rh_raw['330101'] = df_rh['330101']
diff_rh = df_rh - df_rh_raw


# load windspeed data, trim to 2012-present
df_speed = pd.read_csv('{}awn_windspeed_data_5.csv'.format(data_dir), index_col=0)
df_speed = df_speed.loc['2012-01-01 00:00:00':]
df_speed_raw = pd.read_csv('{}awn_windspeed_data_raw.csv'.format(data_dir), index_col=0)
df_speed_raw = df_speed_raw.loc['2012-01-01 00:00:00':]
# fix mt vernon
#df_speed_raw['330101'] = df_speed['330101']



"""
# load windsdir data, trim to 2012-present
df_dir = pd.read_csv('{}awn_winddir_data_2.csv'.format(data_dir), index_col=0) # updated one is 2
df_dir = df_dir.loc['2012-01-01 00:00:00':]
df_dir_raw = pd.read_csv('{}awn_winddir_data_raw.csv'.format(data_dir), index_col=0)
df_dir_raw = df_dir_raw.loc['2012-01-01 00:00:00':]

"""

# load windsdir data, trim to 2012-present
df_dir = pd.read_csv('{}awn_winddir_data_5.csv'.format(data_dir), index_col=0) # updated one is 2
df_dir = df_dir.loc['2012-01-01 00:00:00':]
df_dir_raw = pd.read_csv('{}awn_winddir_data_raw.csv'.format(data_dir), index_col=0)
df_dir_raw = df_dir_raw.loc['2012-01-01 00:00:00':]
# fix mt vernon
#df_dir_raw['330101'] = df_dir['330101']

nas = df_speed.isna().all()
na_cols = nas[nas == True].index
for col in na_cols:
    df_dir[col] = df_dir_raw[col]
    df_speed[col] = df_speed_raw[col]

diff_speed = df_speed - df_speed_raw
diff_dir = df_dir - df_dir_raw

# ------ section -1: wind ------------
total_values_speed = len(df_speed)*len(df_speed.columns)
total_values_dir = len(df_dir)*len(df_dir.columns)
# zeroing out the wind
df_speed_out = pd.DataFrame(index=df_speed.index, columns=df_speed.columns)
speed_out = df_speed.copy().values
df_speed_out[:] = speed_out
df_dir_out = pd.DataFrame(index=df_dir.index, columns=df_dir.columns)
dir_out = df_dir.copy().values
df_dir_out[:] = dir_out

print('speed pct good initially: {}'.format(np.sum(df_speed_out.count())/total_values_speed * 100.))
print('dir pct good initially: {}'.format(np.sum(df_dir_out.count())/total_values_dir * 100.))

# zero out where diffs not equal to 0
speed_out[np.where((diff_speed != 0) | (diff_dir != 0))] = float('nan')
dir_out[np.where((diff_speed != 0) | (diff_dir != 0))] = float('nan')
df_speed_out[:] = speed_out
df_dir_out[:] = dir_out

print('speed pct good manual impute removed: {}'.format(np.sum(df_speed_out.count())/total_values_speed * 100.))
print('dir pct good manual impute removed: {}'.format(np.sum(df_dir_out.count())/total_values_dir * 100.))

# zero out bad temperature data
speed_out[np.where((np.isnan(df_temperature) == True))] = float('nan')
df_speed_out[:] = speed_out
dir_out[np.where((np.isnan(df_dir) == True))] = float('nan')
df_dir_out[:] = dir_out
print('speed pct bad temperature removed: {}'.format(np.sum(df_speed_out.count())/total_values_speed * 100.))
print('dir pct bad temperature removed: {}'.format(np.sum(df_dir_out.count())/total_values_dir * 100.))


# finally, remove 192 in a row checks
# STD direction < 0.5
# Max speed < 1.0
for col in df_dir_out.columns:
    col_tmp = df_dir_out[col].copy()
    col_tmp[:] = False
    for i in range(0, len(df_dir_out) - 192):
        vals = df_dir_out[col][i:i + 192]
        if vals.count() > 50:
            std = vals.std()
            maxval = vals.max()

            if std < 0.5 or maxval < 1.0:
                #wind_match = df_speed[col][i:i+192]
                #if np.mean(wind_match) < 0.2:
                col_tmp.iloc[i:i+192] = True
    x = df_dir_out[col].iloc[np.where((col_tmp == True))[0]]
    print(len(x))
    df_dir_out[col].iloc[np.where((col_tmp == True))[0]] = float('nan')
    df_speed_out[col].iloc[np.where((col_tmp == True))[0]] = float('nan')

print('speed pct final QC 192 loop removed: {}'.format(np.sum(df_speed_out.count())/total_values_speed * 100.))
print('dir pct final QC 192 loop removed: {}'.format(np.sum(df_dir_out.count())/total_values_dir * 100.))

pdb.set_trace()
# get u,v
df_u_out, df_v_out = wind_speed_dir_to_uv(df_speed_out, df_dir_out)


df_dir_out.to_pickle('{}dir_clean.p'.format(out_dir))
df_speed_out.to_pickle('{}speed_clean.p'.format(out_dir))
df_u_out.to_pickle('{}uwnd_clean.p'.format(out_dir))
df_v_out.to_pickle('{}vwnd_clean.p'.format(out_dir))

pdb.set_trace()
"""


# ------ section 0: rel humidity ------------
# count total values
total_values = len(df_rh)*len(df_rh.columns)
print('len of rh: {}'.format(total_values))

# zeroing out the rh
df_rh_out = pd.DataFrame(index=df_rh.index, columns=df_rh.columns)
rh_out = df_rh.copy().values
df_rh_out[:] = rh_out
print('rh pct good initially: {}'.format(np.sum(df_rh_out.count())/total_values * 100.))
# zero out where rhar not equal to 0
rh_out[np.where((diff_rh != 0))] = float('nan')
df_rh_out[:] = rh_out
print('rh pct good manual impute removed: {}'.format(np.sum(df_rh_out.count())/total_values * 100.))
# zero out bad temperature data
rh_out[np.where((np.isnan(df_temperature) == True))] = float('nan')
df_rh_out[:] = rh_out
print('rh pct good bad temperature removed: {}'.format(np.sum(df_rh_out.count())/total_values * 100.))
# zero out T > 90, RH > 90
rh_out[np.where((df_temperature > 90) & (df_rh_out > 90))] = float('nan')
df_rh_out[:] = rh_out
print('rh T>90 RH>90 removed: {}'.format(np.sum(df_rh_out.count())/total_values * 100.))
# zero out SOLAR > 90, RH > 90
rh_out[np.where((df_sol > 1000) & (df_rh_out > 90))] = float('nan')
df_rh_out[:] = rh_out
# zero out RH > 105 or RH < 0
rh_out[np.where((df_rh_out > 105) | (df_rh_out < 0))] = float('nan')
df_rh_out[:] = rh_out
print('rh < 0 or > 105 removed : {}'.format(np.sum(df_rh_out.count())/total_values * 100.))
# finally, remove where RH is nearly unchanged for 192 records in a row
# this is a bit tricky--have to create tmp var for each one with NA values, then NA at the end of that instance in loop
for col in df_rh_out.columns:
    col_tmp = df_rh_out[col].copy()
    col_tmp[:] = False
    for i in range(0, len(df_rh_out) - 192):
        vals = df_rh_out[col][i:i + 192]
        if vals.count() > 50:
            std = vals.std()
            if std < 0.01:
                #wind_match = df_speed[col][i:i+192]
                #if np.mean(wind_match) < 0.2:
                col_tmp.iloc[i:i+192] = True
    x = df_rh_out[col].iloc[np.where((col_tmp == True))[0]]
    print(len(x))
    df_rh_out[col].iloc[np.where((col_tmp == True))[0]] = float('nan')

print('rh stuck sensor removed removed : {}'.format(np.sum(df_rh_out.count())/total_values * 100.))

df_rh_out.to_pickle('{}rh_clean.p'.format(out_dir))


# ------ section 1: solar ------------

# count total values
total_values = len(df_sol)*len(df_sol.columns)
print('len of solar: {}'.format(total_values))

# zeroing out the solar
df_sol_out = pd.DataFrame(index=df_sol.index, columns=df_sol.columns)
sol_out = df_sol.copy().values
df_sol_out[:] = sol_out
print('solar pct good initially: {}'.format(np.sum(df_sol_out.count())/total_values * 100.))
# zero out where solar not equal to 0
sol_out[np.where((diff_sol != 0))] = float('nan')
df_sol_out[:] = sol_out
print('solar pct good manual impute removed: {}'.format(np.sum(df_sol_out.count())/total_values * 100.))
# zero out bad temperature data
sol_out[np.where((np.isnan(df_temperature) == True))] = float('nan')
df_sol_out[:] = sol_out
print('solar pct good bad temperature removed: {}'.format(np.sum(df_sol_out.count())/total_values * 100.))

df_sol_out.to_pickle('{}solar_clean.p'.format(out_dir))


pdb.set_trace()

diff.index = pd.to_datetime(diff.index)
diff = diff.resample('H').mean()
xs, ys = np.meshgrid(diff.index, diff.columns)
vals = np.transpose(diff.values)
fig = plt.figure()
fig.set_size_inches(16, 12)
ax = fig.add_subplot(1, 1, 1)
meshplot = ax.pcolormesh(xs, ys, vals, cmap='seismic', vmin=-1000, vmax=1000)

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.15)
cbar = plt.colorbar(meshplot, cax, extend='both')
cbar.ax.set_ylabel('Difference', fontsize=24)
cbar.ax.tick_params(labelsize=24)
ax.set_xlabel('', fontsize=3)
ax.set_ylabel('Station ID', fontsize=24)
ax.tick_params(axis='x', which='major', labelsize=24)
ax.tick_params(axis='y', which='major', labelsize=6)

ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y'))

plt.savefig('./testplots/solar_difference_raw.png', dpi=300)

pdb.set_trace()


#--------
save_dir = '/data/awn/impute/awn-impute/data/'
df_t_all.to_pickle('{}full_t_raw.p'.format(save_dir))
df_t1_all.to_pickle('{}full_t1_raw.p'.format(save_dir))

pdb.set_trace()


df_final = df_t_all.copy().values


# Change 1: NA where C differs from T
df_final[np.where((df_final != df_c_all))] = float('nan')
# Change 2: NA where C in NA
df_final[np.where((np.isnan(df_c_all) == True))] = float('nan')
# Change 3: NA where major outliers occur
df_final[np.where((df_final > 120))] = float('nan')
df_final[np.where((df_final < -20))] = float('nan')
# Change 4: NA where T and T1 differ by more than 2 F

pdb.set_trace()

"""
