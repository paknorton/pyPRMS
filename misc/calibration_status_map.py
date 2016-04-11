#!/usr/bin/env python2.7

#      Author: Parker Norton (pnorton@usgs.gov)
# Create date: 2016-04-05
# Description: Displays the status of PRMS calibration by HRU jobs on a map

# Although this software program has been used by the U.S. Geological Survey (USGS),
# no warranty, expressed or implied, is made by the USGS or the U.S. Government as
# to the accuracy and functioning of the program and related program material nor
# shall the fact of distribution constitute any such warranty, and no
# responsibility is assumed by the USGS in connection therewith.

# mpl.use('Agg')
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import argparse
import glob
import numpy as np
import os
from osgeo import ogr
import pandas as pd
import prms_cfg

parser = argparse.ArgumentParser(description='Display map showing PRMS calibration status')
parser.add_argument('-c', '--config', help='Primary basin configuration file', required=True)
parser.add_argument('-r', '--runid', help='Runid of the calibration to display', required=True)
parser.add_argument('-s', '--shapefile', help='Filename and path of shapefile for the map', required=True)
parser.add_argument('-k', '--key', help='Shapefile record key for HRUs', default='hru_id_loc')
parser.add_argument('-e', '--extent', help='Extent of shapefile to show', nargs=4, 
                    metavar=('minLon', 'maxLon', 'minLat', 'maxLat'), required=False)

args = parser.parse_args()

configfile = args.config
runid = args.runid
shpfile = args.shapefile
shape_key = args.key

# Name of config file used for each run
basinConfigFile = 'basin.cfg'

# Get a Layer's Extent
extent = args.extent

if not extent or len(extent) != 4:
    print('Using extent information from shapefile')

    # Use gdal/ogr to get the extent information
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(shpfile, 0)
    inLayer = inDataSource.GetLayer()
    extent = inLayer.GetExtent()

east, west, south, north = extent
print('\tExtent: (%f, %f, %f, %f)' % (north, south, east, west))

# Name of shapefile
# shpfile='/media/scratch/PRMS/notebooks/nhru_10U/nhru_10U_simpl'

# Name of attribute to use
# shape_key='hru_id_loc'
# shape_key='hru_id_reg'


# --------------------------------------
# No edits needed below here

cfg = prms_cfg.cfg(configfile)
base_calib_dir = cfg.base_calib_dir

cblabel = 'Calibration Status'
missing_color = '#00BFFF'  # for missing values

# Read the basins_file. Single basin per line
# bfile = open(cfg.get_value('basins_file'), "r")
# basins = bfile.read().splitlines()
# bfile.close()
basins = cfg.get_basin_list()
print('Total basins:', len(basins))

# Check if the runid exists for the first basin
if not os.path.exists('%s/%s/runs/%s' % (base_calib_dir, basins[0], runid)):
    print("ERROR: runid = %s does not exist." % runid)
    exit(1)

# success list
s_list = [el for el in glob.glob('%s/*/runs/%s/.success' % (base_calib_dir, runid))]
s_list += [el for el in glob.glob('%s/*/runs/%s/.warning' % (base_calib_dir, runid))]
s_list += [el for el in glob.glob('%s/*/runs/%s/.error' % (base_calib_dir, runid))]
s_list += [el for el in glob.glob('%s/*/runs/%s/.retry' % (base_calib_dir, runid))]

b_list = []
stat = {'.success': 1, '.warning': 2, '.error': 3, '.retry': 4}
for ee in s_list:
    tmp = ee.split('/')
    ri = tmp.index('runs')
    hru = int(tmp[ri-1].split('_')[1]) + 1

    b_list.append([hru, stat[tmp[-1]]])
    
df = pd.DataFrame(b_list, columns=['HRU', 'status'])
df.sort_values(by=['HRU'], inplace=True)
df.set_index(['HRU'], inplace=True)

# Subset the data
Series_data = df.iloc[:]

fig, ax = plt.subplots(1, figsize=(20, 30))
ax.set_title('Calibration status\n%s\nRUNID = %s' % (base_calib_dir, runid))

print("Loading basemap...")
# Load the basemap
m = Basemap(llcrnrlon=west, llcrnrlat=south, urcrnrlon=east, urcrnrlat=north, resolution='c',
            projection='laea', lat_0=(south+north)/2, lon_0=(east+west)/2, ax=ax)

# Draw parallels.
m.drawparallels(np.arange(0., 90, 10.), labels=[1, 0, 0, 0], fontsize=20)

# Draw meridians
m.drawmeridians(np.arange(180., 360., 10.), labels=[0, 0, 0, 1], fontsize=20)
m.drawmapboundary()

# ------------------------------------------------------------------
# Use basemap to read and draw the shapefile
# Two variables are added to the basemap, m.nhruDd and m.nhruDd_info
#     print 'Loading shapefile...'
m.readshapefile(os.path.splitext(shpfile)[0], 'nhruDd', drawbounds=False)

max_val = 4.
min_val = 0.

# print 'Color HRUs...'
# m.nhruDd contains the lines of the borders
# m.nhruDd_info contains the info on the hru, like the name
for nhruDd_borders, nhruDd_info in zip(m.nhruDd, m.nhruDd_info):
    index = nhruDd_info[shape_key]

    # skip those that aren't in the dataset without complaints
    if index in df.index:
        # set the color for each region
        val = df.loc[index].values[0]

        if pd.isnull(val):
            # Record exists but the value is NaN
            color = missing_color
        elif val == 1:
            color = '#008000'
        elif val == 2:
            color = '#FF8C00'
        elif val == 3:
            color = '#FF0000'
        elif val == 4:
            color = '#FF1493'
    else:
        # The record is totally missing
        color = '#C0C0C0'

    # extract the x and y of the countours and plot them
    xx, yy = zip(*nhruDd_borders)
    patches = ax.fill(xx, yy, facecolor=color, edgecolor='grey')

# Generate a synthetic colorbar starting from the maximum and minimum of the dataset
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", "2%", pad="3%")

# Create a colormap of discrete colors
cmap = mpl.colors.ListedColormap(['#c0c0c0', '#008000', '#ff8c00', '#ff0000', '#FF1493'])
bounds = [0, 1, 2, 3, 4, 5]

cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, boundaries=bounds, spacing='uniform', ticks=[0.5, 1.5, 2.5, 3.5, 4.5])

# Add colorbar tick label text, provides a lot of control over placement
cb1.ax.get_yaxis().set_ticks([])
thelabels = ['Pending/Running', 'Success', 'Warning', 'Error', 'Cancelled']

for j, lab in enumerate(thelabels):
    cb1.ax.text(.5, (2 * j + 1) / float(len(thelabels)*2), lab, ha='center', va='center', weight='bold', color='white', rotation=90)

# Adjust the placement of the colorbar text
# cb1.ax.get_yaxis().labelpad = -20

# Change the tick labels and rotate them
# Doesn't seem to provide control over centering vertically and horizontally
# cb1.ax.set_yticklabels(['Pending/Running', 'Success', 'Warning', 'Error'], va='center', ha='left', rotation=90)

cb1.set_label(cblabel)
ax.patch.set_facecolor('0.93')
    
plt.show()
