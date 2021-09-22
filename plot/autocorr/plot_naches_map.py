'''
Plot site locations
'''
import pdb
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from mpl_toolkits.basemap import maskoceans
import numpy as np
import pandas as pd
#from netCDF4 import Dataset
import time, datetime
from time import gmtime, localtime, strftime
import os

fs1 = 15
fs2 = 15
lw1 = 0.9


indir = '/data/awn/oper/clearwest/data/'
infile = indir+'topo_arrays_final.npz'
indata = np.load(infile)

lons = indata['lons']
lats = indata['lats']
topoin = indata['topoin']

lons2 = [-120.65873, -120.61554, -120.70406]
lats2 = [46.70438, 46.69688,  46.66392]
names = ['Naches', 'Gleed', 'Cowiche']
symbols = ['UL', 'LR', 'LR']
topoin = indata['topoin']
df = pd.DataFrame(columns=['lat', 'lon', 'name', 'symbol'])
df['name'] = names
df['lat'] = lats2
df['lon'] = lons2
df['symbol'] = symbols

#custom colormap
from matplotlib.colors import LinearSegmentedColormap
cMap = []
for value, colour in zip([0,75,275,425,575,1000],['#809980','#80B280','#EAD480','#E0C08C','#BAA38C','#FFFFFF']):
    cMap.append((value/1000.0, colour))
JoeTerrainFade = LinearSegmentedColormap.from_list("custom", cMap)

fig = plt.figure()
fig.set_size_inches(fs1,fs2)
ax = fig.add_axes([0.1,0.1,0.8,0.8])

map = Basemap(projection='cass',resolution='h',
	      lon_0 = -120.65,
	      lat_0 = 46.68,
	      width = 17000,
	      height = 17000,ax=ax)

nx = int((map.xmax-map.xmin)/100.)+1; ny = int((map.ymax-map.ymin)/100.)+1

#convert to meters
topoin = topoin *3.28084

topodat = map.transform_scalar(topoin,lons,lats,nx,ny)
#underwater = np.where((topodat < 0))[0]
#topodat[underwater] = -500

# plot image over map with imshow.
im = map.imshow(topodat,JoeTerrainFade,vmin = 1000, vmax = 3000) #Meters conversion * 0.3048

#Filled coastline
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch


map.drawstates()
map.drawcounties()

"""
# draw parallels.
parallels = np.arange(0.,90,0.5)
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=20, linewidth=1) #0 hides them
# draw meridians
meridians = np.arange(180.,360.,0.5)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=20, linewidth=1)
"""

for i in range(0, len(df.index)):
    xnp, ynp = map(df.lon[i], df.lat[i])
    marker = 'o'
    mc = 'white'

    f9 = map.scatter(xnp, ynp, marker=marker, color=mc, s=275, edgecolor='k', zorder=10, label=None, lw=2)
    if df.symbol[i] == 'UR':
        xnp2, ynp2 = map(df.lon[i]+0.01, df.lat[i]+0.01)
        plt.text(xnp2, ynp2, df.name[i], fontsize=25, ha='left', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'CR':
        xnp2, ynp2 = map(df.lon[i]+0.03, df.lat[i]-0.01)
        plt.text(xnp2, ynp2, df.name[i], fontsize=12, ha='left', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'LR':
        xnp2, ynp2 = map(df.lon[i]+0.005, df.lat[i]-0.002)
        plt.text(xnp2, ynp2, df.name[i], fontsize=35, ha='left', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'LLR':
        xnp2, ynp2 = map(df.lon[i]+0.01, df.lat[i]-0.045)
        plt.text(xnp2, ynp2, df.name[i], fontsize=12, ha='left', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'LLR2':
        xnp2, ynp2 = map(df.lon[i]+0.015, df.lat[i]-0.032)
        plt.text(xnp2, ynp2, df.name[i], fontsize=12, ha='left', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'UL':
        xnp2, ynp2 = map(df.lon[i]-0.005, df.lat[i]+0.0025)
        plt.text(xnp2, ynp2, df.name[i], fontsize=35, ha='right', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'CL':
        xnp2, ynp2 = map(df.lon[i]-0.03, df.lat[i]-0.01)
        plt.text(xnp2, ynp2, df.name[i], fontsize=12, ha='right', va='bottom', color='k', fontweight='bold')
    if df.symbol[i] == 'LL':
        xnp2, ynp2 = map(df.lon[i]-0.005, df.lat[i]-0.0025)
        plt.text(xnp2, ynp2, df.name[i], fontsize=35, ha='right', va='bottom', color='k', fontweight='bold')



cbar = map.colorbar(im,location='right',pad="2%", size="3%", extend='both')
cbar.set_label('Elevation (ft)', fontsize=20)
cbar.ax.tick_params(labelsize=20)



plt.savefig('naches_map_v2.png', bbox_inches='tight',dpi=150)

pdb.set_trace()
