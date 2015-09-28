#!/usr/bin/env python

import prms_lib as prms
# import efc_lib as efc
import calendar
import datetime
import numpy as np
import pandas as pd

def dparse(*dstr):
    dint = [int(x) for x in dstr]

    if len(dint) == 2:
        # For months we want the last day of each month
        dint.append(calendar.monthrange(*dint)[1])
    if len(dint) == 1:
        # For annual we want the last day of the year
        dint.append(12)
        dint.append(calendar.monthrange(*dint)[1])

    return datetime.datetime(*dint)

def dparse_HL(yr, mo, dy):
    # Date parser for working with the date format from PRMS files

    # Convert to integer first
    yr, mo, dy = [int(x) for x in [yr, mo, dy]]
    dt = datetime.datetime(yr, mo, dy)
    return dt


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
    ds_c['wyear'] = np.where(ds_c['month']>9, ds_c['wyear']+1, ds_c['wyear'])
    
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

def pbias(data):
    # Compute the percent bias between simulated and observed
    if 'obs_var' in data.columns:
        # single observed values
        # TODO: write this part
        pass
    else:
        # Assume we're working with ranges of values

        # If all the values in the dataframe are zero then return bias=0
        if (data==0).all().all():
            return 0

        data.loc[:,'closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        data.loc[:,'closest'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['closest'])
        data.loc[:,'closest'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['closest'])

        data['diff'] = 0.
        data['diff'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['diff'])
        data['diff'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['diff'])

        num = data['diff'].sum()
        den = data['closest'].sum()

        return (num/den) * 100.


def nrmse(data):
    # Compute the Normalized Root Mean Square error
    # NRMSE == 0 perfect fit (obs = sim)
    # NRMSE > 1  simulated as good as the average value of all observed data
    if 'obs_var' in data.columns:
        # Single observed values
        os_diff = data['obs_var'] - data['sim_var']
        lt_mean = data['obs_var'].mean(axis=0)

        sq_os_diff = os_diff**2
        sq_olt_diff = (data['obs_var'] - lt_mean)**2

    else:
        # We assume we're working with ranges of values
        # columns must be obs_lower and obs_upper

        # If all the values in the dataframe are zero then return NRMSE=0
        if (data==0).all().all():
            return 0

        data.loc[:,'closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        data.loc[:,'closest'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['closest'])
        data.loc[:,'closest'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['closest'])

        data['diff'] = 0.
        data['diff'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['diff'])
        data['diff'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['diff'])
        lt_mean = ((data['obs_upper'] + data['obs_lower']) / 2.0).mean(axis=0)

        sq_os_diff = data['diff']**2
        sq_olt_diff = (data['closest'] - lt_mean)**2

    return np.sqrt(sq_os_diff.sum() / sq_olt_diff.sum())


def nrmse_fcn(data, obsvar, simvar):
    # Compute the Normalized Root Mean Square error
    # NOTE: intended for private use only
    os_diff = data[obsvar] - data[simvar]
    lt_mean = data[obsvar].mean(axis=0)

    sq_os_diff = os_diff**2
    sq_olt_diff = (data[obsvar] - lt_mean)**2

    return np.sqrt(sq_os_diff.sum() / sq_olt_diff.sum())


def nashsutcliffe(data, obsvar, simvar):
    # Compute Nash-Sutcliffe
    # NS == 1 perfect fit between observed and simulated data
    # NS == 0 fit is as good as using the average value of all observed data

    if 'obs_var' in data.columns:
        # Single observed values
        numerator = np.sum((data[obsvar] - data[simvar])**2)
        lt_mean = data[obsvar].mean(axis=0)
        denominator = np.sum((data[obsvar] - lt_mean)**2)
        ns = 1 - numerator / denominator
    else:
        # Assume we are working with a range of values
        # columns must be named 'obs_lower' and 'obs_upper'
        data['closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        data['closest'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['closest'])
        data['closest'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['closest'])

        numerator = np.sum((data[obsvar] - data[simvar])**2)
        lt_mean = ((data['obs_upper'] + data['obs_lower']) / 2.0).mean(axis=0)
        #ltmean = data[obsvar].mean(axis=0)
        denominator = np.sum((data[obsvar] - lt_mean)**2)
        ns = 1 - numerator / denominator


    #print 'NS:', ns
    # MOCOM only minimizes error so multiply NS by -1
    return ns * -1.


def flowduration(ts):
    """Compute the flow duration for a given dataset"""
    # See http://pubs.usgs.gov/sir/2008/5126/section3.html 
    # for the approach used to compute the flow duration
    # NOTE: This routine expects a pandas TimeSeries
    # TODO: assert error if ts is not a time series

    # We only want valid values, sort the values in descending order
    rankedQ = sorted(ts[ts.notnull()], reverse=True)
    print rankedQ[0:10]


    # Compute the exceedence probability for each Q
    prob = np.arange(len(rankedQ), dtype=np.float_) + 1.0
    prob = 100 * (prob / (len(rankedQ) + 1.0))

    # Return a dataframe of the flow duration / exceedence probability
    return pd.DataFrame({'exceedence': prob, 'Q': rankedQ}, columns=['exceedence', 'Q'])



# =============================================================================
# These lists of functions must be defined after the functions they reference
of_fcn = {'NRMSE': nrmse, 'NS': nashsutcliffe, 'PBIAS': pbias}
#          'NRMSE_FD': nrmse_flowduration}
# =============================================================================

def compute_objfcn(objfcn, data):
    return of_fcn[objfcn](data)


# =============================================================================
def main():
    import argparse
    # import pandas as pd
    # import collections
    import prms_cfg as cfg
    # from addict import Dict

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    cfg_in = cfg.cfg('/Users/pnorton/USGS/Projects/National_Hydrology_Model/code/prms_calib/basin.cfg')

    # Command line arguments
    parser = argparse.ArgumentParser(description='Objective function processing')
    parser.add_argument('data', help='Simulated data')
    #parser.add_argument('obsvar', help='Observed variable')
    #parser.add_argument('simvar', help='Simulate variable')
    parser.add_argument('mocomstats', help='Output stats file for MOCOM')
    parser.add_argument('-d', '--daterange',
                        help='Date range for statistics (YYYY-MM-DD YYYY-MM-DD)',
                        nargs=2, metavar=('stDate', 'enDate'), required=True)
    parser.add_argument('-o', '--output', help='Human-readable stats filename')

    args = parser.parse_args()

    # Get the start and end dates
    st = datetime.datetime(*(map(int, args.daterange[0].split('-'))))
    en = datetime.datetime(*(map(int, args.daterange[1].split('-'))))

    # Get the simulated and observed data filenames
    datafile = args.data

    # Load the simulation data
    sim_data = prms.statvar(datafile).data
    sim_data = sim_data[st:en]

    print '='*40
    print 'Read statvar data'

    # Load the statvar dataframe
    # Range files from Lauren use -99.0 as missing, other files use -999.0
    missing = [-999.0, -99.0]

    # Equate objfcn values to columns and order expected in the data file
    colnm_lookup = {'range': ['obs_lower', 'obs_upper'],
                    'value': ['obs_val'],
                    'daily': ['year', 'month', 'day'],
                    'monthly': ['year', 'month'],
                    'annual': ['year'],
                    'mnmonth': ['month']}

    objfcn_link = cfg_in.get_value('of_link')
    of_dict = cfg_in.get_value('objfcn')


    for kk, vv in objfcn_link.iteritems():
        of_result = 0
        # Each object function link can use one or more objective functions weighted together
        # print '-'*70
        # print '-- outer loop'
        # print kk, vv

        for ii, of in enumerate(vv['of_names']):
            curr_of = of_dict[of]

            # print '-'*50
            # print '\tInner loop'
            # Compute OF for each OF that is part of the current of_link key
            # print '\t', of, curr_of

            # Get the total number of columns for the dtype and obs_intv and build the names to use for the dataframe.
            thecols = []
            thecols.extend(colnm_lookup[curr_of['obs_intv']])
            thecols.extend(colnm_lookup[curr_of['obs_type']])

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Read in the observation values/ranges
            if curr_of['obs_intv'] == 'mnmonth':
                # The index won't be a datetime, instead it's a month value
                df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                                  header=None, na_values=missing, names=thecols, index_col=0)
            else:
                # NOTE: When parsing year-month dates pandas defaults to the 21st of each month. I'm not sure yet
                #       if this will cause a problem.
                #       Annual dates are parsed as Jan-1 of the given year.
                # TODO: if 'obsfile' == statvar then read the observed values in from the statvar file
                df1 = pd.read_csv(curr_of['obs_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                                  header=None, na_values=missing, date_parser=dparse,
                                  names=thecols, parse_dates={'thedate': colnm_lookup[curr_of['obs_intv']]}, index_col='thedate')

                if curr_of['obs_intv'] == 'monthly':
                    df1 = df1.resample('M', how='mean')

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Merge simulated with observed; resample simulated if necessary
            if curr_of['obs_intv'] == 'daily':
                df1_join_sim = df1.join(sim_data.loc[:,curr_of['sim_var']], how='left')
            else:
                if curr_of['obs_intv'] == 'monthly':
                    if curr_of['sim_var'] in ['hru_actet']:
                        # This is for variables that should be summed instead of averaged
                        # FIXME: make this dynamic - maybe embed in basin.cfg?
                        tmp = sim_data.loc[:,curr_of['sim_var']].resample('M', how='sum')
                    else:
                        tmp = sim_data.loc[:,curr_of['sim_var']].resample('M', how='mean')
                elif curr_of['obs_intv'] == 'mnmonth':
                    monthly = sim_data.loc[:,curr_of['sim_var']].resample('M', how='mean')
                    tmp = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
                elif curr_of['obs_intv'] == 'annual':
                    tmp = sim_data.loc[:,curr_of['sim_var']].resample('A-SEP', how='mean')
                else:
                    print "ERROR"
                    exit()
                df1_join_sim = df1.join(tmp, how='left')

            df1_join_sim.rename(columns = {curr_of['sim_var']: 'sim_var'}, inplace=True)

            # =================================================================
            # Read in the subdivide data, if specified
            if curr_of['sd_file'] is not None:
                # The subdivide file must be a daily timestep
                thecols = ['year', 'month', 'day', 'sdval']

                # Read the subdivide data
                df2 = pd.read_csv(curr_of['sd_file'], sep=r"\s*", engine='python', usecols=range(0,len(thecols)),
                                  header=None, na_values=missing,
                                  names=thecols, parse_dates={'thedate': ['year', 'month', 'day']}, index_col='thedate')

                # Merge the subdivide data with the observed data
                if curr_of['obs_intv'] != 'daily':
                    # The observed data is not a daily timestep (subdivide data is daily) so raise an error.
                    print 'ERROR: observed data must be daily timestep when using subdivide data'
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

                df_final = df_final.resample('M', how='mean')
            elif curr_of['of_intv'] == 'annual':
                # We only want to include complete water years
                df_final = get_complete_wyears(df_final)

                # TODO: For now the annual interval is assumed to be water-year based
                df_final = df_final.resample('A-SEP', how='mean')
            elif curr_of['of_intv'] == 'mnmonth':
                # We only want to include complete months
                df_final = get_complete_months(df_final)

                monthly = df_final.resample('M', how='mean')
                df_final = monthly.resample('M', how='mean').groupby(monthly.index.month).mean()
            elif curr_of['of_intv'] in months:
                # We are working with a single month over the time period
                df_final = df_final[df_final.index.month==(months.index(curr_of['of_intv'])+1)]

                # TODO: strip rows with NaN observations out of dataframe
            df_final = df_final.dropna(axis=0, how='any', thresh=None, inplace=False).copy()

            # ** objective function looks for sim_val for simulated and either obs_val or obs_lower, obs_upper
            of_result += vv['of_wgts'][ii] * compute_objfcn(curr_of['of_stat'], df_final)
        # **** for of in vv['of_names']:

        print '%s: %0.6f' % (vv['of_desc'], of_result)
    # **** for kk, vv in objfcn_link.iteritems():


if __name__ == '__main__':
    main()
