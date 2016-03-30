#!/usr/bin/env python

# Process SWE datasets into individual HRUs for a given region for use
# by the PRMS model calibration by HRU.

from __future__ import print_function
# from builtins import range
# from six import iteritems

import pandas as pd
import datetime
import sys

# Regions in the National Hydrology Model
regions = ('01', '02', '03', '04', '05', '06', '07', '08',
           '09', '10L', '10U', '11', '12', '13', '14', '15',
           '16', '17', '18')

# Number of HRUs in each region
hrus_by_region = (2462, 4827, 9899, 5936, 7182, 2303, 8205, 4449,
                  1717, 8603, 10299, 7373, 7815, 1958, 3879, 3441,
                  2664, 11102, 5837)


def load_snodas_swe(filename, st_date, en_date, missing_val=(-9999., 'MEAN(m)')):
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

    # ds1 = pd.read_csv(filename, na_values=missing_val, header=True)
    # ds1.drop(0, inplace=True)
    #
    # ds1.rename(columns={ds1.columns[0]:'thedate'}, inplace=True)
    # ds1['thedate'] = pd.to_datetime(ds1['thedate'])
    # ds1.set_index('thedate', inplace=True)

    # Resample to monthly
    ds1_mth = ds1.resample('M', how='mean')
    ds1_mth = ds1_mth[st_date:en_date]

    # Convert meters to inches
    ds1_mth *= 39.3701
    return ds1_mth


def load_mwbm_swe(filename, st_date, en_date, missing_val=(-9999.0)):
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
    if not isinstance(missing_val, tuple):
        missing_val = tuple(missing_val)

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    ds2 = pd.read_csv(filename, sep=' ', skipinitialspace=True,
                      parse_dates={'thedate': [0, 1]}, index_col=['thedate'])

    # The wb.swe file has messed up column headers after 9999 for r10U, so renumber them.
    ds2.rename(columns=lambda x: ds2.columns.get_loc(x)+1, inplace=True)

    # Pandas messes up the day of the month when parsing dates using just a year and month.
    # Resampling to month again will fix the dates without altering the values.
    ds2 = ds2.resample('M', how='mean')
    ds2 = ds2[st_date:en_date]

    # Convert from millimeters to inches
    ds2 *= 0.0393701
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

    print("Loading SWE:")

    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    # When using the CONUS file we have to pull the region from it
    region_start = sum(hrus_by_region[0:regions.index(region)]) + 1
    region_end = region_start + hrus_by_region[regions.index(region)]

    # Build the set of columns to load
    # Column 0 is the date and is always included
    column_set = [0]
    column_set.extend(range(region_start, region_end))

    print("\tSNODAS SWE..")
    ds1 = pd.read_csv('%s/SNODAS_SWE_CONUS_2003-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                      index_col='thedate', usecols=column_set)

    print("\tMWBM SWE..")
    ds2 = pd.read_csv('%s/MWBM_SWE_CONUS_2003-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                      index_col='thedate', usecols=column_set)


    # print("\tSNODAS SWE..")
    # ds1 = load_snodas_swe('%s/SNODAS_SWE.r%s' % (src_dir, region), st_date, en_date)
    #
    # print("\tMWBM SWE..")
    # ds2 = load_mwbm_swe('%s/mwbm_swe_r%s' % (src_dir, region), st_date, en_date)

    print("Writing out HRUs:")
    for hh in range(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        modis = pd.DataFrame(ds1.ix[:, hh])
        modis.rename(columns={modis.columns[0]: 'modis'}, inplace=True)

        wb = pd.DataFrame(ds2.ix[:, hh])
        wb.rename(columns={wb.columns[0]: 'mwbm'}, inplace=True)

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
    print('')


def pull_by_hru_GCPO(src_dir, dst_dir, st_date, en_date, region):
    # For a given region pull SWE for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir    Location of the AET datasets
    # dstdir    Top-level location to write HRUs
    # st_date   Start date for output dataset
    # en_date   End date for output datasdet
    # region    The region to pull HRUs out of

    # Override the global region information for the GCPO
    regions = ['GCPO']
    hrus_by_region = [20251]

    # if not isinstance(st_date, datetime.datetime):
    #     date_split = st_date.split('-')
    #     st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])
    #
    # if not isinstance(en_date, datetime.datetime):
    #     date_split = en_date.split('-')
    #     en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])
    #
    # # Get the zero-based start and end index for the selected region
    # start_idx = sum(hrus_by_region[0:regions.index(region)])
    # end_idx = (sum(hrus_by_region[0:regions.index(region)]) + hrus_by_region[regions.index(region)] - 1)

    print("Loading SWE:")

    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    print("\tSNODAS SWE..")
    swe_snodas = pd.read_csv('%s/SNODAS_SWE_GCPO_2003-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                             index_col='thedate')

    print("\tMWBM SWE..")
    swe_mwbm = pd.read_csv('%s/MWBM_SWE_GCPO_2003-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                           index_col='thedate')

    print("Writing out HRUs:")
    for hh in range(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        modis = pd.DataFrame(swe_snodas.iloc[:, hh])
        modis.rename(columns={modis.columns[0]: 'modis'}, inplace=True)

        mwbm = pd.DataFrame(swe_mwbm.iloc[:, hh])
        mwbm.rename(columns={mwbm.columns[0]: 'mwbm'}, inplace=True)

        ds_join = modis.join(mwbm)

        ds_join['min'] = ds_join.min(axis=1)
        ds_join['max'] = ds_join.max(axis=1)
        ds_join.drop(['modis', 'mwbm'], axis=1, inplace=True)
        ds_join['year'] = ds_join.index.year
        ds_join['month'] = ds_join.index.month
        ds_join.reset_index(inplace=True)
        ds_join.drop(['thedate'], axis=1, inplace=True)

        # Write out the dataset
        outfile = '%s/r%s_%06d/SWEerror' % (dst_dir, region, hh)
        ds_join.to_csv(outfile, sep=' ', float_format='%0.5f', columns=['year', 'month', 'min', 'max'],
                       header=False, index=False)
    print('')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split MWBM model output into individual HRUs')
    parser.add_argument('-b', '--basedir', help='Base directory for regions', required=True)
    parser.add_argument('-s', '--srcdir', help='Source data directory', required=True)
    parser.add_argument('-r', '--region', help='Region to process', required=True)
    parser.add_argument('--range', help='Create error range files', action='store_true')

    args = parser.parse_args()

    selected_region = args.region
    base_dir = args.basedir
    src_dir = args.srcdir
    dst_dir = '%s/r%s_byHRU' % (base_dir, args.region)

    # selected_region = 'GCPO'
    # src_dir = '/media/scratch/PRMS/datasets/SWE'
    # dst_dir = '/media/scratch/PRMS/regions/r%s_byHRU' % selected_region

    st = datetime.datetime(2003, 10, 31)
    en = datetime.datetime(2010, 12, 31)

    if selected_region == 'GCPO':
        pull_by_hru_GCPO(src_dir, dst_dir, st, en, selected_region)
    else:
        pull_by_hru(src_dir, dst_dir, st, en, selected_region)


if __name__ == '__main__':
    main()
