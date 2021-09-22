"""
autocorrelation
"""

import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from datetime import timedelta

#naches 310126
# gleed 310111
# cowiche 310137

df_temp = pd.read_pickle('/data/awn/impute/awn-impute/data/clean/temp_clean.p')
df_temp.index = pd.to_datetime(df_temp.index)
stn70 = pd.read_pickle('/data/awn/impute/paper/plot/stns_70pct.p')
stn70 = stn70.drop('330101')
ind70 = stn70.index.astype('int')
df_temp = df_temp[stn70.index]
df_temp.index = df_temp.index + timedelta(hours=8)
df_temp = (df_temp-32) * 5/9.

lag15 = df_temp.shift(-1)
lag60 = df_temp.shift(-4)

abs_diff15 = (df_temp-lag15).abs()
abs_diff60 = (df_temp-lag60).abs()
diff15 = (lag15 - df_temp)
diff60 = (lag60 - df_temp)

diff15 = diff15['310126'] #naches
lag15_n = lag15['310126']
lag60_n = lag60['310126']
temp_n = df_temp['310126']
lag15_g = lag15['310111']
temp_g = df_temp['310111']
lag15_c = lag15['310137']
temp_c = df_temp['310137']

diurnal = pd.DataFrame(index=np.linspace(0,23,24), columns=['autocorr','autocorr60','gleed','cowiche','mean', 'absmean','pct25', 'pct75', 'pct05', 'pct95', 'pct01', 'pct99'])

for i in range(0,24):
    l15 = lag15_n[lag15_n.index.hour == i]
    l60 = lag60_n[lag60_n.index.hour == i]
    ttt = temp_n[temp_n.index.hour == i]
    ttg = temp_g[temp_g.index.hour == i]
    ttc = temp_c[temp_c.index.hour == i]
    diurnal['autocorr60'][i] = ttt.corr(l60)
    diurnal['gleed'][i] = ttt.corr(ttg)
    diurnal['cowiche'][i] = ttt.corr(ttc)
    diurnal['autocorr'][i] = ttt.corr(l15)
    diurnal['mean'][i] = np.mean(np.mean(diff15.loc[diff15.index.hour == i]))
    diurnal['absmean'][i] = np.mean(np.mean(abs_diff15.loc[abs_diff15.index.hour == i]))
    diurnal['pct05'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 5))
    diurnal['pct25'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 25))
    diurnal['pct75'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 75))
    diurnal['pct01'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 1))
    diurnal['pct95'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 95))
    diurnal['pct99'][i] = np.mean(np.nanpercentile(diff15.loc[diff15.index.hour == i], 99))

fig = plt.figure()
fig.set_size_inches(5,6)
ax1 = fig.add_subplot(2,1,2)
ax2 = fig.add_subplot(2,1,1)
axes=[ax1,ax2]

meshplot = ax2.plot(diurnal.index+0.5, diurnal['autocorr'],  linestyle='-', color='black', label='Naches - Lag 15-min')
meshplot = ax2.plot(diurnal.index+0.5, diurnal['autocorr60'],  linestyle='-', color='grey', label='Naches -Lag 60-min')
meshplot2 = ax2.plot(diurnal.index+0.5, diurnal['gleed'],  linestyle='-', color='blue', label='Naches - Gleed')
meshplot3 = ax2.plot(diurnal.index+0.5, diurnal['cowiche'],  linestyle='-', color='purple', label='Naches - Cowiche')

ax2.legend(fontsize=7.5)
ax2.set_ylabel('Correlation coefficient ($\it{r}$)')
#ax1.set_xlabel('Hour (UTC)')

ax2.set_xlim(0,24)
ax2.set_ylim(0.96,1)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

#plt.savefig('naches_autocorr.png', dpi=300, bbox_inches='tight')
#plt.close()



diurnal = diurnal.astype('float')

meshplot = ax1.plot(diurnal.index+0.5, diurnal['mean'],  linestyle='-', color='black', label='Mean')
meshplot2 = ax1.plot(diurnal.index+0.5, diurnal['pct01'],  linestyle='-', color='black', label=None, lw=0.4)
meshplot3 = ax1.plot(diurnal.index+0.5, diurnal['pct05'],  linestyle='-', color='black', label=None, lw=0.4)
meshplot4 = ax1.plot(diurnal.index+0.5, diurnal['pct95'],  linestyle='-', color='black', label=None, lw=0.4)
meshplot5 = ax1.plot(diurnal.index+0.5, diurnal['pct99'],  linestyle='-', color='black', label=None, lw=0.4)
ax1.fill_between(diurnal.index+0.5, diurnal['pct01'], diurnal['pct05'], facecolor='blue', alpha=0.3, label='1st/99th PCT')
ax1.fill_between(diurnal.index+0.5, diurnal['pct95'], diurnal['pct99'], facecolor='blue', alpha=0.3, label=None)
ax1.fill_between(diurnal.index+0.5, diurnal['pct05'], diurnal['pct95'], facecolor='blue', alpha=0.6, label='5th/95th PCT')

ax1.legend(fontsize=7.5, loc='lower right')
ax1.set_ylabel('Naches AT$_{t}$ - AT$_{t-15min}$ ($^{\circ}$C)')
ax1.set_xlabel('Hour (UTC)')
ax1.plot([0,24],[0,0], lw=0.3, color='whitesmoke')
#ax1.grid()
#ax2.grid()

ax1.set_xlim(0,24)
ax1.set_ylim(-3,3)

ax1.text(0.03, 0.93,'(b)',
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax1.transAxes, fontsize=12)

ax2.text(0.03, 0.07,'(a)',
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax2.transAxes, fontsize=12)

#ax.tick_params(axis='x', which='major', labelsize=24)
#ax.tick_params(axis='y', which='major', labelsize=6)

plt.savefig('FIG_NACHES_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('FIG_NACHES_v2.eps', bbox_inches='tight')
plt.close()

pdb.set_trace()

