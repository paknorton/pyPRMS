#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as dates
from matplotlib.dates import YearLocator
# from matplotlib.dates import DayLocator, MonthLocator, YearLocator
# from matplotlib.ticker import ScalarFormatter

import prms_cfg
import prms_lib as prms
from prms_calib_helpers import get_sim_obs_data

import argparse
import pandas as pd
import os


def set_ylim(subplot, dataset, padding):
    # Set the y-axis limits
    # Dataset is assumed to be a single column pandas dataset
    maxy = dataset.max()
    miny = dataset.min()
    
    # Shut off automatic offsets for the y-axis
    subplot.get_yaxis().get_major_formatter().set_useOffset(False)

    # Set the limits on the y-axis
    yrange = maxy - miny
    ypad = abs(yrange * padding)
    subplot.set_ylim([miny - ypad, maxy + ypad])


parser = argparse.ArgumentParser(description='Display map showing PRMS calibration status')
parser.add_argument('-b', '--bestruns', help='CSV file of best calibration run ids', required=True)
parser.add_argument('-c', '--config', help='Primary basin configuration file', required=True)
parser.add_argument('-p', '--pdf', help='PDF output filename', required=True)
parser.add_argument('-r', '--runid', help='Runid of the calibration to display', required=True)

args = parser.parse_args()

configfile = args.config
runid = args.runid

# Name of config file used for each run
basinConfigFile = 'basin.cfg'

# ### Plot observed versus simulated monthly and daily streamflow
# Setup model run information
# runid = '2016-04-05_1017'
# basinid = 'rPipestem_002280'
# configfile = '/media/scratch/PRMS/calib_runs/pipestem_1/%s/basin.cfg' % basinid

cfg = prms_cfg.cfg(configfile)

base_calib_dir = cfg.base_calib_dir
limits_file = cfg.param_range_file
template_dir = cfg.get_value('template_dir')

# workdir = '%s/%s/runs/%s' % (base_calib_dir, basinid, runid)

print('base_calib_dir:', base_calib_dir)
print('limits_file:', limits_file)
# print('workdir:', workdir)

# Get the list of basins for this calibration
basins = cfg.get_basin_list()
print('Total basins:', len(basins))

title_size = 13    # Font size for the plot titles
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

cdir = os.getcwd()
print('Starting directory:', cdir)

# pdf_filename = '/media/scratch/PRMS/notebooks/multivars_byhru_%s.pdf' % runid
# pdf_filename = '%s/%s/pdf/%s_%s_%s_statvar.pdf' % (basedir, basinid, basinid, runid, modelrunid)
outpdf = PdfPages(args.pdf)

df = pd.read_csv(args.bestruns, index_col=0)
df_bestrun = df[df['best'] == 'OF_comp'].copy()
df_bestrun.reset_index(inplace=True)
df_bestrun.drop(['index'], axis=1, inplace=True)
df_bestrun.set_index('HRU', inplace=True)
print(df_bestrun.head())

base_dir = cfg.base_dir

for bb in basins:
    print(bb)
    workdir = '{0:s}/{1:s}/{2:s}'.format(base_dir, runid, bb)
    basin_config_file = '{0:s}/{1:s}'.format(workdir, basinConfigFile)
    basin_cfg = prms_cfg.cfg(basin_config_file)

    # workdir = '{0:s}/{1:s}/runs/{2:s}'.format(cfg.base_calib_dir, bb, runid)
    # basin_cfg = prms_cfg.cfg('{0:s}/{1:s}/{2:s}'.format(cfg.base_calib_dir, bb, basinConfigFile))
    # basin_config_file = '{0:s}/{1:s}'.format(workdir, basinConfigFile)

    # TODO: Check for .success file before including an HRU
    if os.path.isfile('{0:s}/.success'.format(workdir)):
        topbarcolor = 'green'
    elif os.path.isfile('{0:s}/.warning'.format(workdir)):
        topbarcolor = 'orange'
    elif os.path.isfile('{0:s}/.error'.format(workdir)):
        topbarcolor = 'red'
        continue
    else:
        continue

    hrunum = int(bb.split('_')[1]) + 1

    csoln = '{0:05d}'.format(df_bestrun[df_bestrun.index == hrunum]['soln_num'].values)
    print('({0:d}) For {1:s} best solution is: {2:s}'.format(hrunum, 'OF_comp', csoln))

    soln_workdir = '{0:s}/{1:s}'.format(workdir, csoln)
    try:
        os.chdir(soln_workdir)
    except:
        # Always want to end up back where we started
        print('Awww... crap!')
        os.chdir(cdir)

    # Override the start_date so all data is plotted, not just the calibration data
    basin_cfg.update_value('start_date', '1981-10-01')
    st_date_calib = prms.to_datetime(basin_cfg.get_value('start_date_model'))
    en_date = prms.to_datetime(basin_cfg.get_value('end_date'))

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(17, 11), sharex=True)
    ax = axes.flatten()
    cc = 0

    objfcn_link = basin_cfg.get_value('of_link')

    for kk, vv in iteritems(objfcn_link):
        for ii, of in enumerate(vv['of_names']):
            curr_of = cfg.get_value('objfcn')[of]
            df_final = get_sim_obs_data(basin_cfg, curr_of)

            # ==================================================================
            # PLOT stuff here
            
            # Plot success/warning/error bar at top
            minx = df_final.index.min().to_pydatetime()
            maxx = df_final.index.max().to_pydatetime()
            
            if 'obs_var' in df_final.columns:
                # Working with individual observations
                # Plot obs/sim plot
                # maxy = df_final['obs_var'].max()
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_var'], color='grey',
                                    label=curr_of['obs_var'])
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['sim_var'], color='red',
                                    label=curr_of['sim_var'])
            else:
                # Working with observation ranges
                # maxy = df_final['obs_upper'].max()
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_upper'],
                                    linewidth=1.0, color='#3399cc', label='{0:s} upper'.format(curr_of['obs_var']))
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_lower'], 'k--',
                                    linewidth=1.0, color='#3399cc', label='{0:s} lower'.format(curr_of['obs_var']))
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['sim_var'], color='red',
                                    label=curr_of['sim_var'])

            miny, maxy = ax[cc].get_ylim()
            ax[cc].plot([minx, maxx], [maxy, maxy], linewidth=10.0, color=topbarcolor, alpha=0.5)
            ax[cc].set_title('Simulated v. observed', fontsize=10)
            
            ax[cc].xaxis.set_major_locator(YearLocator(base=1, month=1, day=1))
            # ax[cc].xaxis.set_minor_locator(MonthLocator(bymonth=[1,4,7,10], bymonthday=1))
            # ax[cc].get_xaxis().set_minor_formatter(dates.DateFormatter('%b'))
            ax[cc].get_xaxis().set_major_formatter(dates.DateFormatter('\n%Y'))

            ax[cc].set_xlim([st_date_calib, en_date])
            ax[cc].legend(loc='upper left', framealpha=0.5)
            
            plt.suptitle('basin: {0:s}\nHRU: {1:d}\nrunid: {2:s} ({3:s})'.format(bb, hrunum, runid, csoln),
                         fontsize=title_size)
            cc += 1
            # Create a secondary y-axis
            # ax2 = ax[cc].twinx()
            # ax2.set_xlim([st, en])

            # Set the secondary y-axis limit
            # set_ylim(ax2, plot_data['basin_ppt'], 0.02)

            # Plot precipitation as a series of vertical lines
            # ax2.vlines(plot_data.index.to_pydatetime(), [0], plot_data['basin_ppt'], color='blue', alpha=0.4)
            # ax2.invert_yaxis()
            
        # **** for of in vv['of_names']:
    outpdf.savefig()
os.chdir(cdir)
outpdf.close()
# plt.show()
