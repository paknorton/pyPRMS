#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

# import calendar
# import datetime
import numpy as np
import pandas as pd

# import prms_lib as prms

# def dparse(*dstr):
#     dint = [int(x) for x in dstr]
#
#     if len(dint) == 2:
#         # For months we want the last day of each month
#         dint.append(calendar.monthrange(*dint)[1])
#     if len(dint) == 1:
#         # For annual we want the last day of the year
#         dint.append(12)
#         dint.append(calendar.monthrange(*dint)[1])
#
#     return datetime.datetime(*dint)


# def dparse_HL(yr, mo, dy):
#     # Date parser for working with the date format from PRMS files
#
#     # Convert to integer first
#     yr, mo, dy = [int(x) for x in [yr, mo, dy]]
#     dt = datetime.datetime(yr, mo, dy)
#     return dt


# def get_complete_months(ds, obsvar=None):
#     # Steps to remove "bad" months from dataset
#     # "bad" is defined as any month where one or more days are missing
#     ds_c = ds.copy()
#
#     # Create a year/month field
#     ds_c['yrmo'] = ds_c.index.map(lambda x: 100*x.year + x.month)
#
#     # Get a list of all year/months which contain missing observations
#     if obsvar is not None:
#         b = ds_c[ds_c[obsvar].isnull()]['yrmo'].tolist()
#     elif 'obs_var' in ds.columns:
#         b = ds_c[ds_c['obs_var'].isnull()]['yrmo'].tolist()
#     else:
#         # Assume a range of observed values
#         # If obs_lower is NaN then obs_upper will be NaN
#         b = ds_c[ds_c['obs_lower'].isnull()]['yrmo'].tolist()
#
#     # Create set of unique year/months which contain missing values
#     badmonths = {x for x in b}
#
#     # Drop entire year/months which contain those missing values
#     c = ds_c.loc[~ds_c['yrmo'].isin(badmonths)]
#
#     # Strip the 'yrmo' column; return the dataset
#     c.drop(['yrmo'], axis=1, inplace=True)
#
#     return c
#
#
# def get_complete_wyears(ds, obsvar=None):
#     # Steps to remove "bad" years from dataset
#     # "bad" is defined as any year where one or more days are missing
#     ds_c = ds.copy()
#
#     ds_c['wyear'] = ds_c.index.year
#     ds_c['month'] = ds_c.index.month
#     ds_c['wyear'] = np.where(ds_c['month'] > 9, ds_c['wyear']+1, ds_c['wyear'])
#
#     # Get a list of all year/months which contain missing observations
#     if obsvar is not None:
#         b = ds_c[ds_c[obsvar].isnull()]['wyear'].tolist()
#     elif 'obs_var' in ds.columns:
#         b = ds_c[ds_c['obs_var'].isnull()]['wyear'].tolist()
#     else:
#         # Assume a range of observed values
#         # If obs_lower is NaN then obs_upper will be NaN
#         b = ds_c[ds_c['obs_lower'].isnull()]['wyear'].tolist()
#
#     # Create set of unique water years which contain missing values
#     badyears = {x for x in b}
#
#     # Drop water years which contain those missing values
#     c = ds_c.loc[~ds_c['wyear'].isin(badyears)]
#
#     # Strip the 'wyear' and 'month' columns; return the dataset
#     c.drop(['wyear', 'month'], axis=1, inplace=True)
#     return c


def pbias(data):
    # Compute the percent bias between simulated and observed
    # pbias = 100 * sum(sim - obs) / sum(obs)

    # If all the values in the dataframe are zero then return bias=0
    if (data == 0).all().all():
        return 0

    if 'obs_var' in data.columns:
        # single observed values
        data['diff'] = 0.
        data['diff'] = data['sim_var'] - data['obs_var']
        num = data['diff'].sum()
        den = data['obs_var'].sum()
    else:
        # Assume we're working with ranges of values
        data.loc[:, 'closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        # data.loc[:,'closest'] = np.where(data['sim_var'] < data['obs_lower'], data['sim_var'] - data['obs_lower'],
        #                                  data['closest'])
        # data.loc[:,'closest'] = np.where(data['sim_var'] > data['obs_upper'], data['sim_var'] - data['obs_upper'],
        #                                  data['closest'])

        data['diff'] = 0.
        data['diff'] = np.where(data['sim_var'] < data['obs_lower'], data['sim_var'] - data['obs_lower'], data['diff'])
        data['diff'] = np.where(data['sim_var'] > data['obs_upper'], data['sim_var'] - data['obs_upper'], data['diff'])

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
        if (data == 0).all().all():
            return 0

        data.loc[:, 'closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        data.loc[:, 'closest'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'],
                                          data['closest'])
        data.loc[:, 'closest'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'],
                                          data['closest'])

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


def nashsutcliffe(data):
    # Compute Nash-Sutcliffe
    # NS == 1 perfect fit between observed and simulated data
    # NS == 0 fit is as good as using the average value of all observed data

    if 'obs_var' in data.columns:
        # Single observed values
        # noinspection PyTypeChecker
        numerator = np.sum((data['obs_var'] - data['sim_var'])**2)
        lt_mean = data['obs_var'].mean(axis=0)
        # noinspection PyTypeChecker
        denominator = np.sum((data['obs_var'] - lt_mean)**2)
        ns = 1 - numerator / denominator
    else:
        # Assume we are working with a range of values
        # columns must be named 'obs_lower' and 'obs_upper'
        data['closest'] = (data['obs_upper'] + data['obs_lower']) / 2.0
        data['closest'] = np.where(data['sim_var'] < data['obs_lower'], data['obs_lower'] - data['sim_var'], data['closest'])
        data['closest'] = np.where(data['sim_var'] > data['obs_upper'], data['obs_upper'] - data['sim_var'], data['closest'])

        # noinspection PyTypeChecker
        numerator = np.sum(data['closest']**2)
        # numerator = np.sum((data[obsvar] - data[simvar])**2)
        lt_mean = ((data['obs_upper'] + data['obs_lower']) / 2.0).mean(axis=0)
        # ltmean = data[obsvar].mean(axis=0)
        # noinspection PyTypeChecker
        denominator = np.sum((data['closest'] - lt_mean)**2)
        ns = 1 - numerator / denominator
    # print 'NS:', ns
    # MOCOM only minimizes error so multiply NS by -1
    return ns * -1.


def flowduration(ts):
    """Compute the flow duration for a given dataset"""
    # See http://pubs.usgs.gov/sir/2008/5126/section3.html 
    # for the approach used to compute the flow duration
    # NOTE: This routine expects a pandas TimeSeries
    # TODO: assert error if ts is not a time series

    # We only want valid values, sort the values in descending order
    ranked_q = sorted(ts[ts.notnull()], reverse=True)
    print(ranked_q[0:10])

    # Compute the exceedence probability for each Q
    prob = np.arange(len(ranked_q), dtype=np.float_) + 1.0
    prob = 100 * (prob / (len(ranked_q) + 1.0))

    # Return a dataframe of the flow duration / exceedence probability
    return pd.DataFrame({'exceedence': prob, 'Q': ranked_q}, columns=['exceedence', 'Q'])


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
    import prms_cfg
    from prms_calib_helpers import get_sim_obs_stat

    cfg = prms_cfg.cfg('basin.cfg')

    # Command line arguments
    parser = argparse.ArgumentParser(description='Objective function processing')
    parser.add_argument('data', help='Simulated data')
    # parser.add_argument('obsvar', help='Observed variable')
    # parser.add_argument('simvar', help='Simulate variable')
    parser.add_argument('mocomstats', help='Output stats file for MOCOM')
    parser.add_argument('-d', '--daterange',
                        help='Date range for statistics (YYYY-MM-DD YYYY-MM-DD)',
                        nargs=2, metavar=('stDate', 'enDate'), required=True)
    parser.add_argument('-o', '--output', help='Human-readable stats filename')

    args = parser.parse_args()

    tmpfile = open(args.mocomstats, 'w')
    objfcn_link = cfg.get_value('of_link')

    for vv in objfcn_link:
        tmpfile.write('%0.6f ' % get_sim_obs_stat(cfg, vv))
    tmpfile.write('\n')
    tmpfile.close()


if __name__ == '__main__':
    main()
