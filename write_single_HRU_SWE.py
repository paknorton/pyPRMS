#!/usr/bin/env python

# Process SWE datasets into individual HRUs for a given region for use
# by the PRMS model calibration by HRU.

import pandas as pd
import numpy as np
import datetime
import sys

# Regions in the National Hydrology Model
regions = ['01', '02', '03', '04', '05', '06', '07', '08',
           '09', '10L', '10U', '11', '12', '13', '14', '15',
           '16', '17', '18']

# Number of HRUs in each region
hrus_by_region = [2462, 4827, 9899, 5936, 7182, 2303, 8205, 4449, 
                  1717, 8603, 10299, 7373, 7815, 1958, 3879, 3441,
                  2664, 11102, 5837]



def load_snodas_swe(filename, st_date, en_date, missing_val=[-9999., 'MEAN(m)']):
    # ---------------------------------------------------------------------------
    # Process SNODAS SWE 
    #
    # Parameters:
    # filename      Full path of file to read
    # st_date       Start date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # en_date       End date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # missing_val   One or more values used for a missing value
    if not isinstance(missing_val, list):
        missing_val = list(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    ds1 = pd.read_csv(filename, sep=' ', na_values=[-9999.0], 
                      header=None, skiprows=3, skipinitialspace=True,
                      parse_dates={'thedate': [0, 1, 2]}, index_col=['thedate'])
    ds1.rename(columns=lambda x: ds1.columns.get_loc(x)+1, inplace=True)

    #ds1 = pd.read_csv(filename, na_values=missing_val, header=True)
    #ds1.drop(0, inplace=True)
    #
    #ds1.rename(columns={ds1.columns[0]:'thedate'}, inplace=True)
    #ds1['thedate'] = pd.to_datetime(ds1['thedate'])
    #ds1.set_index('thedate', inplace=True)

    # Resample to monthly
    ds1_mth = ds1.resample('M', how='mean')
    ds1_mth = ds1_mth[st_date:en_date]

    # Convert meters to inches
    ds1_mth = ds1_mth * 39.3701
    return ds1_mth


def load_mwbm_swe(filename, st_date, en_date, missing_val=[-9999.0]):
    # ---------------------------------------------------------------------------
    # Process MWBM SWE 
    #
    # Parameters:
    # filename      Full path of file to read
    # st_date       Start date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # en_date       End date for returned dataset.
    #               Can be datetime or a string of form 'YYYY-MM-DD'
    # missing_val   One or more values used for a missing value
    if not isinstance(missing_val, list):
        missing_val = list(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])


    #file2 = 'wb.swe'
    ds2 = pd.read_csv(filename, sep=' ', skipinitialspace=True, 
                      parse_dates={'thedate': [0, 1]}, index_col=['thedate'])

    # The wb.swe file has messed up column headers after 9999 for r10U, so renumber them.
    ds2.rename(columns=lambda x: ds2.columns.get_loc(x)+1, inplace=True)

    # Pandas messes up the day of the month when parsing dates using just a year and month.
    # Resampling to month again will fix the dates without altering the values.
    ds2 = ds2.resample('M', how='mean')
    ds2 = ds2[st_date:en_date]

    # Convert from millimeters to inches
    ds2 = ds2 * 0.0393701
    return ds2


def pull_by_hru(src_dir, dst_dir, st_date, en_date, region):
    # For a given region pull SWE for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir    Location of the AET datasets
    # dstdir    Top-level location to write HRUs
    # st_date   Start date for output dataset
    # en_date   End date for output datasdet
    # region    The region to pull HRUs out of

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    print "Loading SWE:"
    print "\tSNODAS SWE.."
    ds1 = load_snodas_swe('%s/SNODAS_SWE.r%s' % (src_dir, region), st_date, en_date)

    print "\tMWBM SWE.."
    ds2 = load_mwbm_swe('%s/mwbm_swe_r%s' % (src_dir, region), st_date, en_date)

    print "Writing out HRUs:"
    for hh in xrange(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()
        
        modis = pd.DataFrame(ds1.ix[:,hh+1])
        modis.rename(columns={modis.columns[0]: 'modis'}, inplace=True)
        #print modis.head()

        wb = pd.DataFrame(ds2.ix[:,hh+1])
        wb.rename(columns={wb.columns[0]: 'mwbm'}, inplace=True)
        #print wb.head()

        ds_join = modis.join(wb)

        ds_join['min'] = ds_join.min(axis=1)
        ds_join['max'] = ds_join.max(axis=1)
        ds_join.drop(['modis', 'mwbm'], axis=1, inplace=True)
        ds_join['year'] = ds_join.index.year
        ds_join['month'] = ds_join.index.month
        ds_join.reset_index(inplace=True)
        ds_join.drop(['thedate'], axis=1, inplace=True)

        # Wrie out the dataset
        outfile = '%s/r%s_%06d/SWEerror' % (dst_dir, region, hh)
        ds_join.to_csv(outfile, sep=' ', float_format='%0.5f', columns=['year', 'month', 'min', 'max'], 
                       header=False, index=False)
        #print ds_join
    print ''


def main():
    selected_region = '10U'
    src_dir = '/media/scratch/PRMS/datasets/SWE'
    dst_dir = '/media/scratch/PRMS/regions/r%s_byHRU' % selected_region

    st = datetime.datetime(2003,10,31)
    en = datetime.datetime(2010,12,31)

    pull_by_hru(src_dir, dst_dir, st, en, selected_region)


if __name__ == '__main__':
    main()


