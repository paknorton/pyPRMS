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
# from matplotlib.ticker import ScalarFormatter

import prms_cfg
import prms_lib as prms
import prms_objfcn
import mocom_lib as mocom
# import re
import pandas as pd
# import numpy as np
# import math as mth
import datetime
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
    

# ### Plot observed versus simulated monthly and daily streamflow
# Setup model run information
runid = '2016-04-05_1017'
basinid = 'rPipestem_002280'
configfile = '/media/scratch/PRMS/calib_runs/pipestem_1/%s/basin.cfg' % basinid
basinConfigFile = 'basin.cfg'

cfg = prms_cfg.cfg(configfile)

base_calib_dir = cfg.base_calib_dir
limits_file = cfg.param_range_file
template_dir = cfg.get_value('template_dir')

workdir = '%s/%s/runs/%s' % (base_calib_dir, basinid, runid)

print('base_calib_dir:', base_calib_dir)
print('limits_file:', limits_file)
print('workdir:', workdir)

# Get the list of basins for this calibration
basins = cfg.get_basin_list()
print('Total basins:', len(basins))

title_size = 13    # Font size for the plot titles
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Setup model run information
# basedir = '/cxfs/projects/usgs/water/mows/NHM/calib/mocom/test_v1'
# statvar_file = 'daymet.statvar'
# nwis_file = '/cxfs/projects/usgs/water/mows/NHM/calib/gage_master/nwis_sites.tab'

cdir = os.getcwd()
print('Starting directory:', cdir)

pdf_filename = '/media/scratch/PRMS/notebooks/multivars_byhru_%s.pdf' % runid
# pdf_filename = '%s/%s/pdf/%s_%s_%s_statvar.pdf' % (basedir, basinid, basinid, runid, modelrunid)
outpdf = PdfPages(pdf_filename)

for bb in basins:
    print(bb)
    workdir = '%s/%s/runs/%s' % (cfg.base_calib_dir, bb, runid)
    basin_cfg = prms_cfg.cfg('%s/%s/%s' % (cfg.base_calib_dir, bb, basinConfigFile))
    basin_config_file = '%s/%s' % (workdir, basinConfigFile)

    # TODO: Check for .success file before including an HRU
    if os.path.isfile('%s/.success' % workdir):
        topbarcolor = 'green'
    elif os.path.isfile('%s/.warning' % workdir):
        topbarcolor = 'orange'
    elif os.path.isfile('%s/.error' % workdir):
        topbarcolor = 'red'
        continue
    else:
        continue

    hrunum = int(bb.split('_')[1]) + 1
    
    # Read in mocom file
    mocom_log = mocom.opt_log(basin_cfg.get_log_file(runid), basin_config_file)
    objfcns = mocom_log.objfcnNames
    lastgen_data = mocom_log.data[mocom_log.data['gennum'] == mocom_log.lastgen]

    # The minimum composite OF result is used to select the pareto set member
    # for each HRU that will be merge back into the parent region
    lastgen_data.loc[:, 'OF_comp'] = 0.5*lastgen_data['OF_AET'] + 0.4*lastgen_data['OF_SWE'] + \
                                     0.1*lastgen_data['OF_runoff']
    
    # Get the 'best' solution number for OF_comp
    csoln = '%05d' % lastgen_data[lastgen_data['OF_comp'] == lastgen_data['OF_comp'].min()]['soln_num']

    print('(%d) For %s best solution is: %s' % (hrunum, 'OF_comp', csoln))

    soln_workdir = '%s/%s' % (workdir, csoln)
    try:
        os.chdir(soln_workdir)
    except:
        # Always want to end up back where we started
        print('Awww... crap!')
        os.chdir(cdir)

    st_date_calib = datetime.datetime(1982, 10, 1)
#     st_date_calib = prms.to_datetime(basin_cfg.get_value('start_date_model'))
    en_date = prms.to_datetime(basin_cfg.get_value('end_date'))
    
    # Load up the control file and the statvar file
    cobj = prms.control(basin_cfg.get_value('prms_control_file'))
    statvar_file = cobj.get_var('stat_var_file')['values'][0]

    # Load the simulation data
    sim_data = prms.statvar(statvar_file).data
    sim_data = sim_data[st_date_calib:en_date]
    
    # We follow alot of the process used during calibration to create the 
    # simulation and observed dataset for plotting.
    
    # Load the statvar dataframe
    # Range files from Lauren use -99.0 as missing, other files use -999.0
    missing = [-999.0, -99.0]

    # Equate objfcn values to columns and order expected in the data file
    colnm_lookup = {'range': ['obs_lower', 'obs_upper'],
                    'value': ['obs_var'],
                    'daily': ['year', 'month', 'day'],
                    'monthly': ['year', 'month'],
                    'annual': ['year'],
                    'mnmonth': ['month']}

    objfcn_link = basin_cfg.get_value('of_link')
    of_dict = basin_cfg.get_value('objfcn')

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(17, 11), sharex=True)
    ax = axes.flatten()
    cc = 0
    
    for kk, vv in iteritems(objfcn_link):
        for ii, of in enumerate(vv['of_names']):
            curr_of = of_dict[of]
#                 print vv['of_desc']

            # Get the total number of columns for the dtype and obs_intv and build the names to use for the dataframe.
            thecols = []
            thecols.extend(colnm_lookup[curr_of['obs_intv']])
            thecols.extend(colnm_lookup[curr_of['obs_type']])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Read in the observation values/ranges
            if curr_of['obs_intv'] == 'mnmonth':
                # The index won't be a datetime, instead it's a month value
                df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0, len(thecols)),
                                  header=None, na_values=missing, names=thecols, index_col=0)
            else:
                # NOTE: When parsing year-month dates pandas defaults to the 21st of each month. I'm not sure yet
                #       if this will cause a problem.
                #       Annual dates are parsed as Jan-1 of the given year.
                # TODO: if 'obsfile' == statvar then read the observed values in from the statvar file
                df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0, len(thecols)),
                                  header=None, na_values=missing, date_parser=prms_objfcn.dparse,
                                  names=thecols, parse_dates={'thedate': colnm_lookup[curr_of['obs_intv']]},
                                  index_col='thedate')

                if curr_of['obs_intv'] == 'monthly':
                    df1 = df1.resample('M', how='mean')

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Merge simulated with observed; resample simulated if necessary
            if curr_of['obs_intv'] == 'daily':
                df1_join_sim = df1.join(sim_data.loc[:, curr_of['sim_var']], how='left')
            else:
                if curr_of['obs_intv'] == 'monthly':
                    if curr_of['sim_var'] in ['hru_actet']:
                        # This is for variables that should be summed instead of averaged
                        # FIXME: make this dynamic - maybe embed in basin.cfg?
                        tmp = sim_data.loc[:, curr_of['sim_var']].resample('M', how='sum')
                    else:
                        tmp = sim_data.loc[:, curr_of['sim_var']].resample('M', how='mean')
                elif curr_of['obs_intv'] == 'mnmonth':
                    monthly = sim_data.loc[:, curr_of['sim_var']].resample('M', how='mean')
                    tmp = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
                elif curr_of['obs_intv'] == 'annual':
                    tmp = sim_data.loc[:, curr_of['sim_var']].resample('A-SEP', how='mean')
                else:
                    print("ERROR")
                    tmp = None
                    exit(1)
                df1_join_sim = df1.join(tmp, how='left')

            df1_join_sim.rename(columns={curr_of['sim_var']: 'sim_var'}, inplace=True)

            # =================================================================
            # Read in the subdivide data, if specified
            if curr_of['sd_file'] is not None:
                # The subdivide file must be a daily timestep
                thecols = ['year', 'month', 'day', 'sdval']

                # Read the subdivide data
                df2 = pd.read_csv(curr_of['sd_file'], sep=r"\s*", engine='python', usecols=range(0, len(thecols)),
                                  header=None, na_values=missing,
                                  names=thecols, parse_dates={'thedate': ['year', 'month', 'day']}, index_col='thedate')

                # Merge the subdivide data with the observed data
                if curr_of['obs_intv'] != 'daily':
                    # The observed data is not a daily timestep (subdivide data is daily) so raise an error.
                    print('ERROR: observed data must be daily timestep when using subdivide data')
                    exit()

                # Merge statvar and observed data
                df_final = df1_join_sim.join(df2, how='left')

                # Subset to only include values which match 'sdval'
                df_final = df_final[df_final['sdval'] == curr_of['sd_val']]
            else:
                df_final = df1_join_sim

            # -----------------------------------------------------------------
            # Now resample to specified of_intv
            if curr_of['of_intv'] == 'monthly':
                # We only want to include complete months
                df_final = prms_objfcn.get_complete_months(df_final)

                df_final = df_final.resample('M', how='mean')
            elif curr_of['of_intv'] == 'annual':
                # We only want to include complete water years
                df_final = prms_objfcn.get_complete_wyears(df_final)

                # TODO: For now the annual interval is assumed to be water-year based
                df_final = df_final.resample('A-SEP', how='mean')
            elif curr_of['of_intv'] == 'mnmonth':
                # We only want to include complete months
                df_final = prms_objfcn.get_complete_months(df_final)

                monthly = df_final.resample('M', how='mean')
                df_final = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
            elif curr_of['of_intv'] in months:
                # We are working with a single month over the time period
                df_final = df_final[df_final.index.month == (months.index(curr_of['of_intv'])+1)]

                # TODO: strip rows with NaN observations out of dataframe
            df_final = df_final.dropna(axis=0, how='any', thresh=None, inplace=False).copy()

            # ==================================================================
            # PLOT stuff here
            
            # Plot success/warning/error bar at top
            
            minx = df_final.index.min().to_pydatetime()
            maxx = df_final.index.max().to_pydatetime()
            
            if 'obs_var' in df_final.columns:
                # Working with individual observations
                # Plot obs/sim plot
                maxy = df_final['obs_var'].max()
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_var'], color='grey',
                                    label=curr_of['obs_var'])
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['sim_var'], color='red',
                                    label=curr_of['sim_var'])
            else:
                # Working with observation ranges
                maxy = df_final['obs_upper'].max()
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_upper'], 
                                    linewidth=1.0, color='#3399cc', label='%s upper' % curr_of['obs_var'])
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['obs_lower'], 'k--',
                                    linewidth=1.0, color='#3399cc', label='%s lower' % curr_of['obs_var'])
                stuff = ax[cc].plot(df_final.index.to_pydatetime(), df_final['sim_var'], color='red',
                                    label=curr_of['sim_var'])
            
            ax[cc].plot([minx, maxx], [maxy, maxy], linewidth=5.0, color=topbarcolor, alpha=0.5)
            ax[cc].set_title('Simulated v. observed', fontsize=10)
            
            ax[cc].xaxis.set_major_locator(YearLocator(base=1, month=1, day=1))
            # ax[cc].xaxis.set_minor_locator(MonthLocator(bymonth=[1,4,7,10], bymonthday=1))
            # ax[cc].get_xaxis().set_minor_formatter(dates.DateFormatter('%b'))
            ax[cc].get_xaxis().set_major_formatter(dates.DateFormatter('\n%Y'))

            ax[cc].set_xlim([st_date_calib, en_date])
            ax[cc].legend(loc='upper right', framealpha=0.5)
            
            plt.suptitle('basin: %s\nHRU: %d\nrunid: %s (%s)' % (bb, hrunum, runid, csoln), fontsize=title_size)
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
plt.show()
