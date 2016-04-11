#!/usr/bin/env python2.7

from __future__ import print_function

import matplotlib as mpl

# mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Create the colormap
# cmap = 'YlGnBu_r'
# cmap = 'GnBu_r' # for snow
# cmap = 'OrRd'  # for liquid
cmap = 'seismic'
# cmap = ['Green', 'W','Sienna']

# create the colormap if a list of names is given, otherwise use the given colormap
lscm = mpl.colors.LinearSegmentedColormap
if isinstance(cmap, (list, tuple)):
    cmap = lscm.from_list('mycm', cmap)
else:
    cmap = plt.get_cmap(cmap)

runid = '2016-04-05_0846'
best_ofs = ['OF_AET', 'OF_SWE', 'OF_runoff', 'OF_comp']
ofs_to_plot = ['OF_AET', 'OF_SWE', 'OF_runoff']

# Load the data to plot
varname = 'Composite'
best_of = 'OF_comp'
of_to_plot = 'OF_AET'
theversion = 1
df = pd.read_csv('/media/scratch/PRMS/notebooks/02_test_and_prototype/%s_best.csv' % runid, index_col=0)

cblabel = 'Percent bias'

missing_color = '#00BFFF'  # for missing values

df2 = df[df['best'] == best_of].copy()
df2.reset_index(inplace=True)
df2.set_index('HRU', inplace=True)
df3 = df2[of_to_plot]
print(df3.head())
# Select the month to plot on the basemap
# Series_data = df.iloc[:,1]
# print df.min().min()
# print df.max().max()


# Name of shapefile
shpfile = '/media/scratch/PRMS/notebooks/shapefiles/upper_pipestem_ll'
# shpfile = '/media/scratch/PRMS/notebooks/nhru_10U/nhru_10U_simpl'

# Name of attribute to use
shape_key = 'hru_id_loc'
# shape_key = 'hru_id_reg'

# Setup output to a pdf file
outpdf = PdfPages('map_pbias_v%d.pdf' % theversion)

# fig, ax = plt.subplots(1,figsize=(20,30))
# ax = plt.gca()
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(30, 20))
ax = axes.flatten()

extent = (47.6, 47, -98.5, -99.9)  # Extent for Pipestem creek watershed
# extent = (50, 42, -95, -114)  # Extent for r10U
north, south, east, west = extent

# Subset the data
# Series_data = df.iloc[:]

for jj, curr_best in enumerate(best_ofs):
    print('curr_best:', curr_best)
    sys.stdout.flush()
    
    df2 = df[df['best'] == curr_best].copy()
    df2.reset_index(inplace=True)
    df2.set_index('HRU', inplace=True)
    
    for ii, curr_of in enumerate(ofs_to_plot):
        print('\tcurr_of:', curr_of)
        sys.stdout.flush()
        axes[jj, ii].set_title('%s Percent Bias for %s' % (curr_of, curr_best))

        df3 = df2[curr_of]
        #     print "Loading basemap..."
        # Load the basemap
        m = Basemap(llcrnrlon=west, llcrnrlat=south, urcrnrlon=east, urcrnrlat=north, resolution='c',
                    projection='laea', lat_0=(south+north)/2, lon_0=(east+west)/2, ax=axes[jj, ii])

        # draw parallels.
        m.drawparallels(np.arange(0., 90, 10.), labels=[1, 0, 0, 0], fontsize=20)

        # draw meridians
        m.drawmeridians(np.arange(180., 360., 10.), labels=[0, 0, 0, 1], fontsize=20)
        m.drawmapboundary()

        # ------------------------------------------------------------------
        # use basemap to read and draw the shapefile
        # Two variables are added to the basemap, m.nhruDd and m.nhruDd_info
        #     print 'Loading shapefile...'
        m.readshapefile(shpfile, 'nhruDd', drawbounds=False)

        # find minimum and maximum of the dataset to normalize the colors
        max_val = 100.
        min_val = -100.
        # max_val = df3.max()
        # min_val = df3.min()

        #     print 'Color HRUs...'
        # m.nhruDd contains the lines of the borders
        # m.nhruDd_info contains the info on the hru, like the name
        for nhruDd_borders, nhruDd_info in zip(m.nhruDd, m.nhruDd_info):
            index = nhruDd_info[shape_key]

            # Skip those that aren't in the dataset without complaints
            if index in df3.index:
                # Set the color for each region
                val = df3.loc[index]

                if pd.isnull(val):
                    # Record exists but the value is NaN
                    color = missing_color
                else:
                    color = cmap((val - min_val) / (max_val - min_val))
            else:
                # The record is totally missing
                color = '#c0c0c0'

            # Extract the x and y of the countours and plot them
            xx, yy = zip(*nhruDd_borders)
            patches = axes[jj, ii].fill(xx, yy, facecolor=color, edgecolor=color)

        # Generate a synthetic colorbar starting from the maximum and minimum of the dataset
        divider = make_axes_locatable(axes[jj, ii])
        cax = divider.append_axes("right", "2%", pad="3%")

        # axc, kw = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb1.set_label(cblabel)
        # cb1.set_ticks()
        # cb1.ax.tick_params(labelsize=20)
        axes[jj, ii].patch.set_facecolor('0.93')
    
outpdf.savefig()
outpdf.close()
# plt.show()
