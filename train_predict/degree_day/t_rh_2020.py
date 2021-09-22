
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
import pdb

# set experiment name and params
test_year = 2020
n_estimators = 20
exp_name = 't_rh_2020'
regressor = 'rf' # rf


def fit_predict_one_station(X, y, test_year):

    X_test = X[X.index.year == test_year]
    X_train  = X[X.index.year != test_year]
    y_test = y[y.index.year == test_year]
    y_train  = y[y.index.year != test_year]

    # drop where y_train is NA
    y_train_na = y_train.index[y_train.isna() == True]
    X_train.drop(index=y_train_na, inplace=True)
    y_train = y_train.drop(index=y_train_na)

    # build regression model. Could adjust n_estimators. Keep max_depth None. There are other hyperparameters you can
    # play with also -- see docs
    if regressor == 'rf':
        regr = RandomForestRegressor(max_depth=None, random_state=0, verbose=2, n_estimators=n_estimators, n_jobs=4)
    if regressor == 'lin':
        regr = LinearRegression()
    # train model
    regr.fit(X_train, y_train)
    # make predictions
    y_predict = regr.predict(X_test)

    df_actual[column] = y_test
    df_predict[column] = y_predict
    # calculate average absolute error
    diff = np.abs(y_predict - y_test)
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
    df_actual.to_pickle('/data/awn/impute/paper/data/train_predict/degree_day/{}_actual.p'.format(exp_name))
    df_predict.to_pickle('/data/awn/impute/paper/data/train_predict/degree_day/{}_predict.p'.format(exp_name))
    df_stats.to_csv('/data/awn/impute/paper/data/train_predict/degree_day/{}_stats.csv'.format(exp_name))
    return


#df_temp.dropna(how='all', inplace=True)
df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_temp.index = pd.to_datetime(df_temp.index)
df_rh = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/rh_clean.p')
df_rh.index = pd.to_datetime(df_rh.index)
#df_solar = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/solar_clean.p')
#df_solar.index = pd.to_datetime(df_solar.index)

# rename rh columns so there is no overlap
for column in df_rh.columns:
    df_rh.rename(columns={column: '{}_rh'.format(column)}, inplace=True)

# combine data here if necessary
df_combined = pd.concat((df_temp, df_rh), axis=1)
#df_combined = df_temp.copy()

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_simple = imp_mean.fit_transform(df_combined)
df_simple = pd.DataFrame(index=df_combined.index, columns=df_combined.columns)
df_simple[:] = X_simple

columns = df_temp.columns

# train-test split based on year this time (faux one to fill arrays)
X_test_tmp = df_simple[df_simple.index.year == test_year]
X_train_tmp = df_simple[df_simple.index.year != test_year]

df_actual = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_predict = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_stats = pd.DataFrame(index=columns, columns=['mean', 'std', 'max', 'pct95', 'pct99', 'pct999', 'time'])

start_time = time.time()

for column in columns:
    # drop the T column from the station that we are predicting for (i.e. don't use station T to predict station T)
    column_rh = '{}_rh'.format(column)
    X = df_simple.drop(columns=[column, column_rh])
    y = df_combined[column]
    fit_predict_one_station(X, y, test_year)



