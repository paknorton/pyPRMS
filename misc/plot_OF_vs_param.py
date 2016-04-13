#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

# Uncomment the following line to write to pdf without opening a window
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

import prms_lib as prms
import prms_cfg
import mocom_lib as mocom
import argparse
import numpy as np
# import os

# print(os.getcwd())

parser = argparse.ArgumentParser(description='Display map showing PRMS calibration status')
parser.add_argument('-c', '--config', help='Primary basin configuration file', required=True)
# parser.add_argument('-p', '--pdf', help='PDF output filename', required=True)
parser.add_argument('-r', '--runid', help='Runid of the calibration to display', required=True)
parser.add_argument('--lastqtr', help='Only plot last 25% of runs', action='store_true', default=False)

args = parser.parse_args()

configfile = args.config
runid = args.runid

cfg = prms_cfg.cfg(configfile)

basinid = cfg.get_value('basin')
base_calib_dir = cfg.base_calib_dir

print('base_calib_dir:', base_calib_dir)

# Plot last 25% of the generations? If false return all
top_qtr = args.lastqtr

# Read in parameter file used for MOCOM calibration
try:
    limits = cfg.get_param_limits()

    params = limits['parameter'].tolist()
    have_limits = True
    # print(limits)
except IOError:
    print("WARNING: %s does not exist.\nLimits will not be plotted." % cfg.param_range_file)
    have_limits = False

# Read input parameter file to get the default parameter values
# Read the initial input parameters values and compute the mean for each one
initial_param_values = prms.parameters('%s/%s/%s' % (cfg.get_value('template_dir'), basinid,
                                                     cfg.get_value('prms_input_file')))

# Build list of initial/default values for the calibration parameters
initvals = []
for vv in params:
    initvals.append(np.mean(initial_param_values.get_var(vv)['values']))

# ### Plot objective-function(s) versus parameters
workdir = '%s/%s/runs/%s' % (base_calib_dir, basinid, runid)
optfile = cfg.get_log_file(runid)

# Plot all the generation results
if top_qtr:
    pdf_filename = '%s/%s/pdf/%s_%s_OF_v_params_lastQTR.pdf' % (base_calib_dir, basinid, basinid, runid)
else:
    pdf_filename = '%s/%s/pdf/%s_%s_OF_v_params.pdf' % (base_calib_dir, basinid, basinid, runid)

mocom_log = mocom.opt_log(optfile, configfile)
objfcns = mocom_log.objfcnNames

print('Last generation:', mocom_log.lastgen)

if top_qtr:
    # Just work with the last 25% of the generations
    log_data = mocom_log.data[mocom_log.data['gennum'] > int(mocom_log.lastgen * 0.75)]
else:
    log_data = mocom_log.data

# Generate the colors array
colors = cm.autumn(np.linspace(0, 1, mocom_log.lastgen*100))
# --------------------------------------------------------------

# Setup output to a pdf file
outpdf = PdfPages(pdf_filename)
numrows = int(round(len(params) / 4. + 0.5))
# print(numrows)

for oo in objfcns:
    miny = min(log_data[oo])
    maxy = max(log_data[oo])
    
    fig, axes = plt.subplots(nrows=numrows, ncols=4, figsize=(20, 5*numrows))
    ax = axes.flatten()

    for ii, pp in enumerate(params):
        # Range of x-axis for parameter
        if have_limits:
            maxx = limits.iloc[ii, 1]
            minx = limits.iloc[ii, 2]
        else:
            maxx = max(log_data[pp])
            minx = min(log_data[pp])
        
        # Shut off automatic offsets for the y-axis
        ax[ii].get_yaxis().get_major_formatter().set_useOffset(False)

        # Set the limits on the x-axis
        xxrange = maxx - minx
        xpad = abs(xxrange * 0.1)
        ax[ii].set_xlim([minx - xpad, maxx + xpad])
        
        # Set the limits on the y-axis
        yrange = maxy - miny
        ypad = abs(yrange * 0.1)
        ax[ii].set_ylim([miny - ypad, maxy + ypad])

        try:
            ax[ii].scatter(log_data[pp], log_data[oo], color=colors, alpha=0.5)
        except KeyError:
            continue
        
        # Plot parameter range limits
        if have_limits:
            ax[ii].plot([maxx, maxx], [miny - ypad, maxy + ypad], 'k--', color='red')
            ax[ii].plot([minx, minx], [miny - ypad, maxy + ypad], 'k--', color='red')
        
        ax[ii].plot([initvals[ii], initvals[ii]], [miny, maxy], markeredgecolor='black', markerfacecolor='yellow',
                    color='grey', marker='D')

        # Re-plot the final pareto set 
        ax[ii].scatter(log_data[pp].loc[log_data['gennum'] == mocom_log.lastgen],
                       log_data[oo].loc[log_data['gennum'] == mocom_log.lastgen], color='black', marker='x')
        ax[ii].set_title(pp, fontsize=12)

    plt.suptitle('Basin: %s (%s)\nObjective Function vs. Parameters\n%s' % (basinid, runid, oo), fontsize=14)
    # plt.subplots_adjust(top=0.75)
    # plt.subplots_adjust(hspace=0.3)
    outpdf.savefig()

outpdf.close()
print('Output written to %s' % pdf_filename)
# plt.show()
