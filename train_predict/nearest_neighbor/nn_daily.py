
"""
1) predict using SIMPLE imputed dataset plus ACTUAL y for each site
2) run as a loop
3) output predictions (y_predict) vs. actual (y_actual)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import time
import geopy.distance
import pdb

# set experiment name and params
train_size = 0.9 # must be 0.9 or lower
n_stations = 1
n_estimators = 20
exp_name = 'nn_daily_tmax_updated'
exp_type = 'nn'


def get_distance_from_station(stid, meta):
    """
    gets the distance to all of the stations in km
    """
    distances = pd.Series(index=columns)
    for column in columns:
        coords_1 = (meta.loc[stid]['lat'], meta.loc[stid]['lon'])
        coords_2 = (meta.loc[column]['lat'], meta.loc[column]['lon'])
        distance = geopy.distance.great_circle(coords_1, coords_2).km
        distances.loc[column] = distance
        distances = distances.sort_values()

    return distances


def fit_predict_one_station(X, y, train_size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # if test size > 0.1, need to cut training dataset by an appropriate amouunt
    if train_size < 0.9:
        # retain length is the fraction of the whole dataset as set by train_size
        # this works because the training data is already randomized
        retain_length = int(np.round(len(y_train) * (train_size / 0.9), 0))
        y_train = y_train[0:retain_length]
        X_train = X_train[0:retain_length]

    # drop where y_train is NA
    y_train_na = y_train.index[y_train.isna() == True]
    X_train.drop(index=y_train_na, inplace=True)
    y_train = y_train.drop(index=y_train_na)

    df_actual[column] = y_test
    df_predict[column] =  X_test#y_test-X_test[X.columns[0]]
    # calculate average absolute error
    diff = np.abs(y_test -X_test[X.columns[0]])
    pct95 = np.percentile(diff.dropna(), 95)
    pct99 = np.percentile(diff.dropna(), 99)
    pct999 = np.percentile(diff.dropna(), 99.9)

    end_time = time.time()
    df_stats['time'][column] = end_time - start_time

    df_stats['mean'][column] = np.mean(diff)
    df_stats['std'][column] = np.std(diff)
    df_stats['max'][column] = np.max(diff)
    df_stats['pct95'][column] = pct95
    df_stats['pct99'][column] = pct99
    df_stats['pct999'][column] = pct999

    # Save the values
    df_actual.to_pickle('/data/awn/impute/paper/data/train_predict/{}/{}_actual.p'.format(exp_type, exp_name))
    df_predict.to_pickle('/data/awn/impute/paper/data/train_predict/{}/{}_predict.p'.format(exp_type, exp_name))
    df_stats.to_csv('/data/awn/impute/paper/data/train_predict/{}/{}_stats.csv'.format(exp_type, exp_name))

    return

# get metadata
meta = pd.read_pickle('/data/awn/impute/paper/train_predict/num_stations/meta_lat_lon.p')

#df_temp.dropna(how='all', inplace=True)
#df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
#df_temp.index = pd.to_datetime(df_temp.index)

daily_dir = '/data/awn/impute/paper/data/daily'
df_tmax = pd.read_pickle('{}/tmax_daily.p'.format(daily_dir))
df_tmin = pd.read_pickle('{}/tmin_daily.p'.format(daily_dir))

stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
stn70 = stn70.drop('330101')
ind70 = stn70.index.astype('int')
df_tmax = df_tmax[stn70.index]
df_tmin = df_tmin[stn70.index]

columns = df_tmax.columns

# CHANGE TO TMIN TO DO TMIN
df_simple = df_tmax.copy()

# do a faux split to get the split size to build the dataframes
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(df_simple, df_simple[columns[0]], test_size=0.1,
                                                                    random_state=42)
df_actual = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_predict = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_stats = pd.DataFrame(index=columns, columns=['mean', 'std', 'max', 'pct95', 'pct99', 'pct999', 'mean_distance', 'time'])

start_time = time.time()

for column in columns:
    # get the distance to other stations:
    distances = get_distance_from_station(column, meta)

    # keep n_stations
    distances = distances[0:n_stations+1]
    df_stats['mean_distance'][column] = np.mean(distances[1:])

    # get keep list:
    keep_list = distances.index.values
    keep_columns = keep_list.copy()

    df_simple_tmp = df_simple[keep_columns]

    # drop the T column from the station that we are predicting for (i.e. don't use station T to predict station T)
    X = df_simple_tmp.drop(columns=[column])
    y = df_simple[column]
    fit_predict_one_station(X, y, train_size)



pdb.set_trace()