"""
Deg day timeseries vs. predict
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

datadir = '/data/awn/impute/paper/data/train_predict/degree_day/'
actual = pd.read_pickle('{}t_rh_2020_actual.p'.format(datadir))
predict = pd.read_pickle('{}t_rh_2020_predict.p'.format(datadir))

stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
stn70 = stn70.drop('330101')
ind70 = stn70.index.astype('int')

actual = actual[stn70.index]
predict = predict[stn70.index]

base_temp = 50.
max_temp = 86.
actual[actual > 86.] = 86.
predict[predict > 86.] = 86.

# make nan values where actual is NA
predict[np.isnan(actual) == True] = np.nan

actual_50 = actual - base_temp
actual_50[actual_50 < 0] = 0
actual_dd50 = actual_50.resample('1D').sum() / 96.
actual_max = actual.resample('1D').max()
actual_min = actual.resample('1D').min()
actual_mean = actual.resample('1D').mean()
actual_mean2 = (actual_max + actual_min) / 2.
actual_dd50_2 = actual_mean2 - base_temp
actual_dd50_2[actual_dd50_2 < 0] = 0

actual_dd50 = actual_dd50.drop(index=datetime(2020,7,9))
actual_dd50_2 = actual_dd50_2.drop(index=datetime(2020,7,9))

predict_50 = predict - base_temp
predict_50[predict_50 < 0] = 0
predict_dd50 = predict_50.resample('1D').sum() / 96.
predict_max = predict.resample('1D').max()
predict_min = predict.resample('1D').min()
predict_mean = predict.resample('1D').mean()
predict_mean2 = (predict_max + predict_min) / 2.
predict_dd50_2 = predict_mean2 - base_temp
predict_dd50_2[predict_dd50_2 < 0] = 0

predict_dd50 = predict_dd50.drop(index=datetime(2020,7,9))
predict_dd50_2 = predict_dd50_2.drop(index=datetime(2020,7,9))

predict_jul1 = predict_dd50.cumsum().loc[datetime(2020,7,1)]
actual_jul1 = actual_dd50.cumsum().loc[datetime(2020,7,1)]

predict_jul1_2 = predict_dd50_2.cumsum().loc[datetime(2020,7,1)]
actual_jul1_2 = actual_dd50_2.cumsum().loc[datetime(2020,7,1)]

diff = predict_jul1 - actual_jul1
diff_2 = predict_jul1_2 - actual_jul1_2
abs_diff = abs(diff)
abs_diff_2 = abs(diff_2)
pct_diff = diff / actual_jul1*100.
pct_diff_2 = diff_2 / actual_jul1_2*100.


# histogram of high vs. low temperature
diff_max = predict_max - actual_max
diff_min = predict_min - actual_min

diff_max_c = (predict_max - actual_max) * 5/9.
diff_min_c = (predict_min - actual_min) * 5/9.

# histogram of percent difference
hist, bin_edges = np.histogram(pct_diff, bins=70, range=[-17.5,17.5], normed=None, weights=None, density=None)
ind = np.linspace(-17.5,17,70)
width=0.4


fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='indigo')
plt.ylabel('Number of stations')
plt.xlabel('Predict - Obs Difference (%)')
plt.xlim(-14,14)
plt.xticks(np.linspace(-14,14,15))
#plt.title('Predict-Actual % difference')
plt.savefig('pct_diff_INTEGRAL_base50_max86.png', bbox_inches='tight',dpi=300)

hist, bin_edges = np.histogram(pct_diff_2, bins=70, range=[-17.5,17.5], normed=None, weights=None, density=None)
width=0.4

fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='indigo')
plt.ylabel('Number of stations')
plt.xlabel('Predict - Obs Difference (%)')
plt.xlim(-14,14)
plt.xticks(np.linspace(-14,14,15))
#plt.title('Predict-Actual % difference')
plt.savefig('pct_diff_2_DAILY_base50_max86.png', bbox_inches='tight',dpi=300)


pdb.set_trace()

fig = plt.figure()
fig.set_size_inches(5.5,6)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
axes=[ax1,ax2]
hist, bin_edges = np.histogram(pct_diff, bins=70, range=[-17.5,17.5], normed=None, weights=None, density=None)
ind = np.linspace(-17.5,17,70)
width=0.4



ax2.set_xlabel('Predict - Obs GDD Difference (%)')
for ax in axes:
    ax.set_xlim(-10, 10)
    ax.set_xticks(np.linspace(-10, 10, 11))



plt.savefig('pct_diff_2panel_base41.png', bbox_inches='tight',dpi=200)


hist, bin_edges = np.histogram(diff_max.values, bins=70, range=[-17.5,17.5], normed=None, weights=None, density=None)
ind = np.linspace(-17.5,17,70)

fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='indigo')
plt.ylabel('Number of days')
plt.xlabel('Predict - Obs Difference (F)')
plt.xlim(-18,18)
plt.xticks(np.linspace(-16,16,9))
#plt.title('Predict-Actual % difference')
plt.savefig('max_temp.png', bbox_inches='tight',dpi=200)

hist, bin_edges = np.histogram(diff_min.values, bins=70, range=[-17.5,17.5], normed=None, weights=None, density=None)
ind = np.linspace(-17.5,17,70)

fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='indigo')
plt.ylabel('Number of days')
plt.xlabel('Predict - Obs Difference (F)')
plt.xlim(-18,18)
plt.xticks(np.linspace(-16,16,9))
#plt.title('Predict-Actual % difference')
plt.savefig('min_temp.png', bbox_inches='tight',dpi=200)

 #---celsius

width = 0.20
hist, bin_edges = np.histogram(diff_max_c.values, bins=40, range=[-5,5], normed=None, weights=None, density=None)
ind = np.linspace(-4.875,4.875,40)

fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='red')
plt.ylabel('Number of days')
plt.xlabel('Predict - Obs Difference ($^{\circ}$C)')
plt.xlim(-4,4)
#plt.title('Predict-Actual % difference')
plt.savefig('max_temp_c.png', bbox_inches='tight',dpi=200)

hist, bin_edges = np.histogram(diff_min_c.values, bins=40, range=[-5,5], normed=None, weights=None, density=None)
ind = np.linspace(-4.875,4.875,40)

fig = plt.figure()
fig.set_size_inches(5.5,3)
p1 = plt.bar(ind+0.25, hist, width, color='blue')
plt.ylabel('Number of days')
plt.xlabel('Predict - Obs Difference ($^{\circ}$C)')
plt.xlim(-4,4)
#plt.title('Predict-Actual % difference')
plt.savefig('min_temp_c.png', bbox_inches='tight',dpi=200)




pdb.set_trace()

# timeseries

fig,ax = plt.subplots(figsize=(9,4))
#ksea.plot(ax=ax)
plt.title('2019 {} Validation'.format(stnname))
ax.plot(dates, df['Day1 Tmax'], '-', color='red', label='1 day Predicted High')
ax.plot(dates, df['Day1 Tmin'], '-', color='blue', label='1 day Predicted Low')
#ax.plot(dates, df['Day2 Tmax'], '-', color='red', alpha=0.6, label='2 day Predicted High')
#ax.plot(dates, df['Day2 Tmin'], '-', color='blue', alpha=0.6, label='2 day Predicted Low')
ax.plot(dates, df['Day3 Tmax'], '-', color='red', alpha=0.3, label='3 day Predicted High')
ax.plot(dates, df['Day3 Tmin'], '-', color='blue', alpha=0.3, label='3 day Predicted Low')
ax.plot(dates, df['Obs Tmax'], '-',  lw=1, color='black', label='Observed')
ax.plot(dates, df['Obs Tmin'], '-', lw=1, color='black')

# Clean up the plot a bit
from matplotlib.dates import DayLocator, DateFormatter
ax.xaxis.set_major_locator(DayLocator(1))
ax.xaxis.set_major_formatter(DateFormatter('%b'))
ax.set_ylabel('Temperature ($^{\circ}$F)')
ax.set_xlim(dates[0], dates[-1]+timedelta(days=1))
#ax.set_ylabel('Date')
ax.grid()
ax.legend(loc='upper left', fontsize=8)

plt.savefig('{}_validation.png'.format(stnid), dpi=200, bbox_inches='tight')







pdb.set_trace()