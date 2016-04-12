#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
from future.utils import itervalues

# import calendar
# import datetime
import numpy as np
import pandas as pd




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
    # import argparse
    import prms_cfg
    from prms_calib_helpers import get_sim_obs_stat

    cfg = prms_cfg.cfg('basin.cfg')

    # Command line arguments
    # parser = argparse.ArgumentParser(description='Objective function processing')
    # parser.add_argument('data', help='Simulated data')
    # parser.add_argument('obsvar', help='Observed variable')
    # parser.add_argument('simvar', help='Simulate variable')
    # parser.add_argument('mocomstats', help='Output stats file for MOCOM')
    # parser.add_argument('-d', '--daterange',
    #                     help='Date range for statistics (YYYY-MM-DD YYYY-MM-DD)',
    #                     nargs=2, metavar=('stDate', 'enDate'), required=True)
    # parser.add_argument('-o', '--output', help='Human-readable stats filename')

    # args = parser.parse_args()

    # tmpfile = open(args.mocomstats, 'w')
    objfcn_link = cfg.get_value('of_link')

    for vv in itervalues(objfcn_link):
        get_sim_obs_stat(cfg, vv)
        # print('%0.6f ' % get_sim_obs_stat(cfg, vv))
        # tmpfile.write('%0.6f ' % get_sim_obs_stat(cfg, vv))
    # tmpfile.write('\n')
    # tmpfile.close()


if __name__ == '__main__':
    main()
