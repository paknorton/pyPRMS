

# Create date: 2015-05-19
# Author: Parker Norton (pnorton@usgs.gov)
# Description: Misc helper functions for calibration.
#              Most likely many of these will get move elsewhere eventually
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import numpy as np
import operator
import pandas as pd
import prms_lib as prms
import prms_objfcn as objfcn

# Related parameters
# These parameters need to satisfy relationships before PRMS is allowed to run
related_params = {'soil_rechr_max': {'soil_moist_max': operator.le},
                  'tmax_allsnow': {'tmax_allrain': operator.lt}}

months = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')

# Equate objfcn values to columns and order expected in the data file
colnm_lookup = {'range': ['obs_lower', 'obs_upper'],
                'value': ['obs_var'],
                'daily': ['year', 'month', 'day'],
                'monthly': ['year', 'month'],
                'annual': ['year'],
                'mnmonth': ['month']}

# These are parameters/variables which should be summed for statistics
accum_vars = ['hru_actet']

# def dparse_HL(yr, mo, dy):
#     # Date parser for working with the date format from PRMS files
#
#     # Convert to integer first
#     yr, mo, dy = [int(x) for x in [yr, mo, dy]]
#     dt = datetime.datetime(yr, mo, dy)
#     return dt


def get_complete_months(ds, obsvar=None):
    # Steps to remove "bad" months from dataset
    # "bad" is defined as any month where one or more days are missing
    ds_c = ds.copy()

    # Create a year/month field
    ds_c['yrmo'] = ds_c.index.map(lambda x: 100*x.year + x.month)

    # Get a list of all year/months which contain missing observations
    if obsvar is not None:
        b = ds_c[ds_c[obsvar].isnull()]['yrmo'].tolist()
    elif 'obs_var' in ds.columns:
        b = ds_c[ds_c['obs_var'].isnull()]['yrmo'].tolist()
    else:
        # Assume a range of observed values
        # If obs_lower is NaN then obs_upper will be NaN
        b = ds_c[ds_c['obs_lower'].isnull()]['yrmo'].tolist()

    # Create set of unique year/months which contain missing values
    badmonths = {x for x in b}

    # Drop entire year/months which contain those missing values
    c = ds_c.loc[~ds_c['yrmo'].isin(badmonths)]

    # Strip the 'yrmo' column; return the dataset
    c.drop(['yrmo'], axis=1, inplace=True)

    return c


def get_complete_wyears(ds, obsvar=None):
    # Steps to remove "bad" years from dataset
    # "bad" is defined as any year where one or more days are missing
    ds_c = ds.copy()

    ds_c['wyear'] = ds_c.index.year
    ds_c['month'] = ds_c.index.month
    ds_c['wyear'] = np.where(ds_c['month'] > 9, ds_c['wyear']+1, ds_c['wyear'])

    # Get a list of all year/months which contain missing observations
    if obsvar is not None:
        b = ds_c[ds_c[obsvar].isnull()]['wyear'].tolist()
    elif 'obs_var' in ds.columns:
        b = ds_c[ds_c['obs_var'].isnull()]['wyear'].tolist()
    else:
        # Assume a range of observed values
        # If obs_lower is NaN then obs_upper will be NaN
        b = ds_c[ds_c['obs_lower'].isnull()]['wyear'].tolist()

    # Create set of unique water years which contain missing values
    badyears = {x for x in b}

    # Drop water years which contain those missing values
    c = ds_c.loc[~ds_c['wyear'].isin(badyears)]

    # Strip the 'wyear' and 'month' columns; return the dataset
    c.drop(['wyear', 'month'], axis=1, inplace=True)
    return c


def read_default_params(filename):
    # Read in the default parameter ranges
    default_rng_file = open(filename, 'r')
    raw_range = default_rng_file.read().splitlines()
    default_rng_file.close()
    it = iter(raw_range)
    
    def_ranges = {}
    it.next()
    for line in it:
        flds = line.split(' ')
        def_ranges[flds[0]] = {'max': float(flds[1]), 'min': float(flds[2])}
    
    return def_ranges


def read_sens_params(filename, include_params=(), exclude_params=()):
    # Read in the sensitive parameters
    
    try:
        sensparams_file = open(filename, 'r')
    except IOError:
        # print "\tERROR: Missing hruSens.csv file for %s... skipping" % bb
        print('\tERROR: Missing %s file... skipping' % filename)
        return {}
        
    rawdata = sensparams_file.read().splitlines()
    sensparams_file.close()
    it = iter(rawdata)

    counts = {}
    for line in it:
        flds = line.split(',')
        for ff in flds:
            ff = ff.strip()

            try:
                int(ff)
            except:
                if ff not in exclude_params:
                    if ff not in counts:
                        counts[ff] = 0
                    counts[ff] += 1
    
    # Add in the include_params if they are missing from the sensitive parameter list
    for pp in include_params:
        if pp not in counts:
            counts[pp] = 0
    return counts


def adjust_param_ranges(paramfile, calib_params, default_ranges, outfilename, make_dups=False):
    """Adjust and write out the calibration parameters and ranges"""
    src_params = prms.parameters(paramfile)

    # Write the param_list file
    outfile = open(outfilename, 'w')
    for kk, vv in iteritems(calib_params):
        # Grab the current param (kk) from the .params file and verify the
        # upper and lower bounds. Modify them if necessary.

        src_vals = src_params.get_var(kk)['values']
        src_mean = np.mean(src_vals)
        src_min = np.min(src_vals)
        src_max = np.max(src_vals)
        
        # Set upper and lower bounds
        user_min = default_ranges[kk]['min']
        user_max = default_ranges[kk]['max']
        if user_min > src_min:
            user_min = src_min
        if user_max < src_max:
            user_max = src_max

        # Adjustment value to prevent zeroes in calculations
        thresh_adj = abs(user_min) + 10.
        
        adjmin = ((user_min + thresh_adj) * (src_mean + thresh_adj) / (src_min + thresh_adj)) - thresh_adj
        adjmax = ((user_max + thresh_adj) * (src_mean + thresh_adj) / (src_max + thresh_adj)) - thresh_adj
                
        if round(adjmin, 5) != round(default_ranges[kk]['min'], 5):
            print('\t%s: lower bound adjusted (%f to %f)' % (kk, default_ranges[kk]['min'], adjmin))
        if round(adjmax, 5) != round(default_ranges[kk]['max'], 5):
            print('\t%s: upper bound adjusted (%f to %f)' % (kk, default_ranges[kk]['max'], adjmax))
        
        if make_dups:
            # Duplicate each parameter by the number of times it occurred
            # This is for a special use case when calibrating individual values of
            # a parameter.
            for dd in range(vv):
                outfile.write('%s %f %f\n' % (kk, adjmax, adjmin))
        else:
            # Output each parameter once
            outfile.write('%s %f %f\n' % (kk, adjmax, adjmin))
    outfile.close()


def get_sim_obs_stat(cfg, of_info, verbose=True):
    """Create a dataframe containing observed and simulated data and compute the objective function
    :type of_info: object function information object
    :type cfg: configuration object
    """

    # NOTE: cfg must be a prms_cfg object

    # Range files from Lauren use -99.0 as missing, other files use -999.0
    missing = [-999.0, -99.0]

    # Get the name of the observation file
    prms_control = prms.control(cfg.get_value('prms_control_file'))
    statvar_file = prms_control.get_var('stat_var_file')['values'][0]

    st_date_calib = prms.to_datetime(cfg.get_value('start_date'))
    en_date = prms.to_datetime(cfg.get_value('end_date'))

    # Load the simulation data
    sim_data = prms.statvar(statvar_file).data
    sim_data = sim_data[st_date_calib:en_date]

    of_dict = cfg.get_value('objfcn')
    of_result = 0

    for ii, of in enumerate(of_info['of_names']):
        curr_of = of_dict[of]

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
                              header=None, na_values=missing, date_parser=prms.dparse,
                              names=thecols, parse_dates={'thedate': colnm_lookup[curr_of['obs_intv']]},
                              index_col='thedate')

            if curr_of['obs_intv'] == 'monthly':
                df1 = df1.resample('M').mean()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Merge simulated with observed; resample simulated if necessary
        if curr_of['obs_intv'] == 'daily':
            df1_join_sim = df1.join(sim_data.loc[:, curr_of['sim_var']], how='left')
        else:
            if curr_of['obs_intv'] == 'monthly':
                if curr_of['sim_var'] in accum_vars:
                    # Variables that should be summed instead of averaged
                    tmp = sim_data.loc[:, curr_of['sim_var']].resample('M').sum()
                else:
                    tmp = sim_data.loc[:, curr_of['sim_var']].resample('M').mean()
            elif curr_of['obs_intv'] == 'mnmonth':
                monthly = sim_data.loc[:, curr_of['sim_var']].resample('M').mean()
                tmp = monthly.resample('M').mean().groupby(monthly.index.month).mean()
            elif curr_of['obs_intv'] == 'annual':
                tmp = sim_data.loc[:, curr_of['sim_var']].resample('A-SEP').mean()
            else:
                print("ERROR: Objective function invterval (%s) is not an acceptable interval." %
                      curr_of['obs_intv'])
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
            df_final = get_complete_months(df_final)

            df_final = df_final.resample('M').mean()
        elif curr_of['of_intv'] == 'annual':
            # We only want to include complete water years
            df_final = get_complete_wyears(df_final)

            # TODO: For now the annual interval is assumed to be water-year based
            df_final = df_final.resample('A-SEP').mean()
        elif curr_of['of_intv'] == 'mnmonth':
            # We only want to include complete months
            df_final = get_complete_months(df_final)

            monthly = df_final.resample('M').mean()
            df_final = monthly.resample('M').mean().groupby(monthly.index.month).mean()
        elif curr_of['of_intv'] in months:
            # We are working with a single month over the time period
            df_final = df_final[df_final.index.month == (months.index(curr_of['of_intv']) + 1)]

            # TODO: strip rows with NaN observations out of dataframe
        df_final = df_final.dropna(axis=0, how='any', thresh=None, inplace=False).copy()
        # print(df_final.head())

        # ** objective function looks for sim_val for simulated and either obs_val or obs_lower, obs_upper
        of_result += of_info['of_wgts'][ii] * objfcn.compute_objfcn(curr_of['of_stat'], df_final)
        # **** for of in vv['of_names']:

        if verbose:
            print('%s: %0.6f' % (of_info['of_desc'], of_result))

        return of_result
        # **** for kk, vv in objfcn_link.iteritems():


def get_sim_obs_data(cfg, curr_of, verbose=True):
    """Create a dataframe containing observed and simulated data and compute the objective function
    :type curr_of: object function
    :type cfg: configuration object
    """

    # NOTE: cfg must be a prms_cfg object

    # Range files from Lauren use -99.0 as missing, other files use -999.0
    missing = [-999.0, -99.0]

    # Get the name of the observation file
    prms_control = prms.control(cfg.get_value('prms_control_file'))
    statvar_file = prms_control.get_var('stat_var_file')['values'][0]

    st_date_calib = prms.to_datetime(cfg.get_value('start_date'))
    st_date_model = prms.to_datetime(cfg.get_value('start_date_model'))
    en_date = prms.to_datetime(cfg.get_value('end_date'))

    # Load the simulation data
    # We load any data within the entire model range
    # The get_sim_obs_stat() function only loads data in the calibration range
    sim_data = prms.statvar(statvar_file).data
    sim_data = sim_data[st_date_model:en_date]

    # of_dict = cfg.get_value('objfcn')

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
                          header=None, na_values=missing, date_parser=prms.dparse,
                          names=thecols, parse_dates={'thedate': colnm_lookup[curr_of['obs_intv']]},
                          index_col='thedate')

        if curr_of['obs_intv'] == 'monthly':
            df1 = df1.resample('M').mean()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Merge simulated with observed; resample simulated if necessary
    if curr_of['obs_intv'] == 'daily':
        df1_join_sim = df1.join(sim_data.loc[:, curr_of['sim_var']], how='left')
    else:
        if curr_of['obs_intv'] == 'monthly':
            if curr_of['sim_var'] in accum_vars:
                # Variables that should be summed instead of averaged
                tmp = sim_data.loc[:, curr_of['sim_var']].resample('M').sum()
            else:
                tmp = sim_data.loc[:, curr_of['sim_var']].resample('M').mean()
        elif curr_of['obs_intv'] == 'mnmonth':
            monthly = sim_data.loc[:, curr_of['sim_var']].resample('M').mean()
            tmp = monthly.resample('M').mean().groupby(monthly.index.month).mean()
        elif curr_of['obs_intv'] == 'annual':
            tmp = sim_data.loc[:, curr_of['sim_var']].resample('A-SEP').mean()
        else:
            print("ERROR: Objective function invterval (%s) is not an acceptable interval." %
                  curr_of['obs_intv'])
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
        df_final = get_complete_months(df_final)

        df_final = df_final.resample('M').mean()
    elif curr_of['of_intv'] == 'annual':
        # We only want to include complete water years
        df_final = get_complete_wyears(df_final)

        # TODO: For now the annual interval is assumed to be water-year based
        df_final = df_final.resample('A-SEP').mean()
    elif curr_of['of_intv'] == 'mnmonth':
        # We only want to include complete months
        df_final = get_complete_months(df_final)

        monthly = df_final.resample('M').mean()
        df_final = monthly.resample('M').mean().groupby(monthly.index.month).mean()
    elif curr_of['of_intv'] in months:
        # We are working with a single month over the time period
        df_final = df_final[df_final.index.month == (months.index(curr_of['of_intv']) + 1)]

        # TODO: strip rows with NaN observations out of dataframe
    df_final = df_final.dropna(axis=0, how='any', thresh=None, inplace=False).copy()

    return df_final
