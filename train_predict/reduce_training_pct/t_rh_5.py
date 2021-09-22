
"""
1) predict using SIMPLE imputed dataset plus ACTUAL y for each site
2) run as a loop
3) output predictions (y_predict) vs. actual (y_actual)
"""

import numpy as np
import pandas as pd
import pdb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import time

# set experiment name and params
train_size = 0.05 # must be 0.9 or lower
n_estimators = 20
exp_name = 't_rh_5'


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

    # build regression model. Could adjust n_estimators. Keep max_depth None. There are other hyperparameters you can
    # play with also -- see docs
    regr = RandomForestRegressor(max_depth=None, random_state=0, verbose=2, n_estimators=n_estimators)
    # train model
    regr.fit(X_train, y_train)
    # make predictions
    y_predict = regr.predict(X_test)

    df_actual[column] = y_test
    df_predict[column] = y_predict
    # calculate average absolute error
    diff = np.abs(y_predict - y_test)
    pct95 = np.percentile(diff.dropna(), 95)

    end_time = time.time()
    df_stats['time'][column] = end_time - start_time

    df_stats['mean'][column] = np.mean(diff)
    df_stats['std'][column] = np.std(diff)
    df_stats['max'][column] = np.max(diff)
    df_stats['pct95'][column] = pct95

    # Save the values
    df_actual.to_pickle('/data/awn/impute/paper/data/train_predict/reduce_training_pct/{}_actual.p'.format(exp_name))
    df_predict.to_pickle('/data/awn/impute/paper/data/train_predict/reduce_training_pct/{}_predict.p'.format(exp_name))
    df_stats.to_csv('/data/awn/impute/paper/data/train_predict/reduce_training_pct/{}_stats.csv'.format(exp_name))

    return


#df_temp.dropna(how='all', inplace=True)
df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_temp.index = pd.to_datetime(df_temp.index)
df_rh = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/rh_clean.p')
df_rh.index = pd.to_datetime(df_rh.index)

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

# do a faux split to get the split size to build the dataframes
X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(df_simple, df_simple[columns[0]], test_size=0.1,
                                                                    random_state=42)
df_actual = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_predict = pd.DataFrame(index=X_test_tmp.index, columns=columns)
df_stats = pd.DataFrame(index=columns, columns=['mean', 'std', 'max', 'pct95', 'time'])

start_time = time.time()

for column in columns:
    # drop the T column from the station that we are predicting for (i.e. don't use station T to predict station T)
    column_rh = '{}_rh'.format(column)
    X = df_simple.drop(columns=[column, column_rh])
    y = df_combined[column]
    fit_predict_one_station(X, y, train_size)


