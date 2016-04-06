#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import os
import pandas as pd
import prms_lib as prms
import prms_cfg
import prms_objfcn
import mocom_lib as mocom

runid = '2016-04-05_0846'
basinConfigFile = 'basin.cfg'
calibConfigFile = '/media/scratch/PRMS/calib_runs/pipestem_1/%s' % basinConfigFile

# Read the calibration configuration file
cfg = prms_cfg.cfg(calibConfigFile)

print(os.getcwd())
print('runid:', runid)

# Get the list of basins for this calibration
basins = cfg.get_basin_list()
print('Total basins: {0:d}'.format(len(basins)))

base_calib_dir = cfg.get_value('base_calib_dir')

merged_df = None
test_group = ['OF_AET', 'OF_SWE', 'OF_runoff', 'OF_comp']
# test_group = ['OF_AET', 'OF_SWE', 'OF_comp']
test_out = []

cdir = os.getcwd()
print('Starting directory: %s' % cdir)

months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

for bb in basins:
    workdir = '%s/%s/runs/%s' % (cfg.base_calib_dir, bb, runid)
    basin_cfg = prms_cfg.cfg('%s/%s/%s' % (cfg.base_calib_dir, bb, basinConfigFile))
    basin_config_file = '%s/%s' % (workdir, basinConfigFile)

    # TODO: Check for .success file before including an HRU
    if not (os.path.isfile('%s/.success' % workdir) or os.path.isfile('%s/.warning' % workdir)):
        continue

    hrunum = int(bb.split('_')[1]) + 1
    
    # Read in mocom file
    mocom_log = mocom.opt_log(basin_cfg.get_log_file(runid), basin_config_file)
    objfcns = mocom_log.objfcnNames
    lastgen_data = mocom_log.data[mocom_log.data['gennum'] == mocom_log.lastgen]

    # The minimum composite OF result is used to select the pareto set member
    # for each HRU that will be merge back into the parent region
    lastgen_data.loc[:, 'OF_comp'] = 0.45*lastgen_data['OF_AET'] + 0.45*lastgen_data['OF_SWE'] + \
                                     0.1*lastgen_data['OF_runoff']
    # lastgen_data.loc[:, 'OF_comp'] = 0.5*lastgen_data['OF_AET'] + 0.5*lastgen_data['OF_SWE']
    
    print(hrunum,)
    for tt in test_group:
        # Get the set with the best NRMSE for the current test_group OF
        # if hrunum == 29:
        #     print '\t', lastgen_data[lastgen_data[tt] == lastgen_data[tt].min()]['soln_num']
        
        try:
            csoln = '%05d' % lastgen_data[lastgen_data[tt] == lastgen_data[tt].min()]['soln_num']
        except: 
            # We probably got multiply matches to the minimum value so 
            # use AET (for SWE), SWE (for AET), or SWE (for runoff)
            # as a tie breaker
            csoln = 0
            print('tie-breaker!', hrunum, tt)
            tmp1 = lastgen_data[lastgen_data[tt] == lastgen_data[tt].min()]
            if tt == 'OF_SWE':
                csoln = '%05d' % tmp1[tmp1['OF_AET'] == tmp1['OF_AET'].min()]['soln_num']
            elif tt == 'OF_AET':
                csoln = '%05d' % tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]['soln_num']
            elif tt == 'OF_runoff':
                csoln = '%05d' % tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]['soln_num']
                
#         print '-'*40
        print('(%d) For %s best solution is: %s' % (hrunum, tt, csoln))

        soln_workdir = '%s/%s' % (workdir, csoln)
        try:
            os.chdir(soln_workdir)
        except:
            # Always want to end up back where we started
            print('Awww... crap!')
            os.chdir(cdir)
        
        st_date_calib = prms.to_datetime(basin_cfg.get_value('start_date'))
        en_date = prms.to_datetime(basin_cfg.get_value('end_date'))

        # ====================================================================
        # 2015-09-28 PAN: taken from prms_post_mocom.py
        # Get the name of the observation file
        cobj = prms.control(basin_cfg.get_value('prms_control_file'))
        statvar_file = cobj.get_var('stat_var_file')['values'][0]

        tmpfile = open("tmpstats", 'w')
        outputstats = []

        # Load the simulation data
        sim_data = prms.statvar(statvar_file).data
        sim_data = sim_data[st_date_calib:en_date]
        # print '='*40
        # print 'Read statvar data'

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

        objfcn_link = cfg.get_value('of_link')
        of_dict = cfg.get_value('objfcn')
        
        tmp_data = []

        tmp_data.append(hrunum)
        tmp_data.append(csoln)
        tmp_data.append(tt)

        for kk, vv in iteritems(objfcn_link):
            of_result = 0

            for ii, of in enumerate(vv['of_names']):
                curr_of = of_dict[of]
#                 print vv['of_desc']
                
                # Replace a couple of entries
                curr_of['of_stat'] = 'PBIAS'
                curr_of['of_intv'] = 'mnmonth'

                # Get the total number of columns for the dtype and obs_intv and build the names to use
                # for the dataframe.
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
                                      names=thecols, parse_dates={'thedate': ['year', 'month', 'day']},
                                      index_col='thedate')

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

#                 if tt == 'OF_runoff' and curr_of['sim_var'] == 'pkwater_equiv':
#                     df_final.plot()
                
                # ** objective function looks for sim_val for simulated and either obs_var or obs_lower, obs_upper
                of_result += vv['of_wgts'][ii] * prms_objfcn.compute_objfcn(curr_of['of_stat'], df_final)
            # **** for of in vv['of_names']:
            tmp_data.append(of_result)
            # print of_result
            
        # Add results to the test_out list
        test_out.append(tmp_data)
    os.chdir(cdir)

# Create dataframe of results
ofnames = mocom_log.objfcnNames
ofnames.insert(0, 'best')
ofnames.insert(0, 'soln_num')
ofnames.insert(0, 'HRU')
xx = pd.DataFrame(test_out, columns=ofnames)

# write to a csv file
xx.to_csv('%s_best.csv' % runid)
xx.head()

# In[ ]:
bb = xx[xx.best == 'OF_AET']
print('Best AET (min/max/median/mean): %0.1f/%0.1f/%0.1f/%0.1f' %
      (bb.OF_AET.min(), bb.OF_AET.max(), bb.OF_AET.median(), bb.OF_AET.mean()))

bb = xx[xx.best == 'OF_SWE']
print('Best SWE (min/max/median/mean): %0.1f/%0.1f/%0.1f/%0.1f' %
      (bb.OF_SWE.min(), bb.OF_SWE.max(), bb.OF_SWE.median(), bb.OF_SWE.mean()))

bb = xx[xx.best == 'OF_comp']
print('Best Composite: SWE (min/max/median/mean): %0.1f/%0.1f/%0.1f/%0.1f' %
      (bb.OF_SWE.min(), bb.OF_SWE.max(), bb.OF_SWE.median(), bb.OF_SWE.mean()))

bb = xx[xx.best == 'OF_comp']
print('Best Composite: AET (min/max/median/mean): %0.1f/%0.1f/%0.1f/%0.1f' %
      (bb.OF_AET.min(), bb.OF_AET.max(), bb.OF_AET.median(), bb.OF_AET.mean()))

# In[ ]:
# lastgen_data[lastgen_data['OF_AET'] == lastgen_data['OF_AET'].min()]
# lastgen_data[lastgen_data['OF_SWE'] == lastgen_data['OF_SWE'].min()]
# lastgen_data[lastgen_data['OF_comp'] == lastgen_data['OF_comp'].min()]

# In[ ]:
# print tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]
# csoln = '%05d' % tmp1[tmp1['OF_SWE'] == tmp1['OF_SWE'].min()]
