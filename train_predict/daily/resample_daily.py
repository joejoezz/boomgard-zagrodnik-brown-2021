"""
resample from 15-min to daily data...NA if more than 3 hours (12 obs) missing
create...
Tmax
Tmin
Total Solar
Mean u-wind
Mean v-wind
"""

import numpy as np
import pandas as pd
import pdb
from datetime import datetime, timedelta, date

def wind_uv_to_speed_dir(uval, vval):
    """
    Converts U and V component of wind to a speed and direction
    """
    vel_val = np.sqrt(uval**2 + vval**2)
    wdir = 180/np.pi * np.arctan2(uval, vval)
    wdir += 180
    wdir[wdir < 0] = wdir[wdir < 0]+360.
    return vel_val, wdir

daily_dir = '/data/awn/impute/paper/data/daily'

start_date = date(2012,1,1)
end_date = date(2020,6,8)
new_index = pd.date_range(start=start_date, end=end_date)

df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_temp.index = pd.to_datetime(df_temp.index)
df_temp = df_temp.drop(columns=['330101'])

df_solar = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/solar_clean.p')
df_solar.index = pd.to_datetime(df_solar.index)
df_solar = df_solar.drop(columns=['330101'])

df_rh = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/rh_clean.p')
df_rh.index = pd.to_datetime(df_rh.index)
df_rh = df_rh.drop(columns=['330101'])

df_uwnd = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/uwnd_clean.p')
df_uwnd.index = pd.to_datetime(df_uwnd.index)
df_uwnd = df_uwnd.drop(columns=['330101'])

df_vwnd = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/vwnd_clean.p')
df_vwnd.index = pd.to_datetime(df_vwnd.index)
df_vwnd = df_vwnd.drop(columns=['330101'])

df_tmax_out = pd.DataFrame(columns=df_temp.columns)
df_tmin_out = pd.DataFrame(columns=df_temp.columns)
df_sol_out = pd.DataFrame(columns=df_solar.columns)
df_rh_out = pd.DataFrame(columns=df_rh.columns)
df_speed_out = pd.DataFrame(columns=df_uwnd.columns)
df_dir_out = pd.DataFrame(columns=df_vwnd.columns)


while start_date <= end_date:
    temp_today = df_temp.loc[df_temp.index.date == start_date]
    nan_sum = temp_today.isna().sum()
    hi_temp_daily = temp_today.max()
    lo_temp_daily = temp_today.min()
    hi_temp_daily[(nan_sum >= 16)] = float('nan')
    lo_temp_daily[(nan_sum >= 16)] = float('nan')
    df_tmax_out.loc[start_date] = hi_temp_daily
    df_tmin_out.loc[start_date] = lo_temp_daily

    rh_today = df_rh.loc[df_rh.index.date == start_date]
    nan_sum = rh_today.isna().sum()
    rh_daily = rh_today.mean()
    rh_daily[(nan_sum >= 16)] = float('nan')
    df_rh_out.loc[start_date] = rh_daily

    sol_today = df_solar.loc[df_solar.index.date == start_date]
    nan_sum = sol_today.isna().sum()
    sol_daily = sol_today.sum()
    sol_daily[(nan_sum >= 16)] = float('nan')
    df_sol_out.loc[start_date] = sol_daily

    uwnd_today = df_uwnd.loc[df_uwnd.index.date == start_date]
    vwnd_today = df_vwnd.loc[df_vwnd.index.date == start_date]
    speed_today, dir_today = wind_uv_to_speed_dir(uwnd_today, vwnd_today)

    nan_sum = speed_today.isna().sum()
    speed_daily = speed_today.mean()
    speed_daily[(nan_sum >= 16)] = float('nan')
    df_speed_out.loc[start_date] = speed_daily

    nan_sum = dir_today.isna().sum()
    dir_daily = dir_today.mean()
    dir_daily[(nan_sum >= 16)] = float('nan')
    df_dir_out.loc[start_date] = dir_daily

    start_date += timedelta(days=1)
    print(start_date)

df_tmax_out.to_pickle('{}/tmax_daily.p'.format(daily_dir))
df_tmin_out.to_pickle('{}/tmin_daily.p'.format(daily_dir))
df_rh_out.to_pickle('{}/rh_daily.p'.format(daily_dir))
df_sol_out.to_pickle('{}/sol_daily.p'.format(daily_dir))
df_speed_out.to_pickle('{}/speed_daily.p'.format(daily_dir))
df_dir_out.to_pickle('{}/dir_daily.p'.format(daily_dir))

pdb.set_trace()

