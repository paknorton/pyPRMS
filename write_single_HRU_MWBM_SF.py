#!/usr/bin/env python

# Process streamflow from the Monthly Water Balance Model (MWBM) into individual
# HRUs for a given region for use by PRMS model calibration by HRU.

import prms_objfcn
import prms_lib as prms
import pandas as pd
import numpy as np
import datetime
import sys
# The runoff error bounds from the Monthly Water Balance Model (MWBM) are
# located on yeti at /cxfs/projects/usgs/water/mows/MWBMRUNSwGRPparams/MWBMerr.
# The runoff is broken out by region with a single file for each min and max set of values.
# Runoff values are in units of mm/day. To convert from mm/day to cubic feet per second the
# following conversion is used.
# 

# Regions in the National Hydrology Model
regions = ['01', '02', '03', '04', '05', '06', '07', '08',
           '09', '10L', '10U', '11', '12', '13', '14', '15',
           '16', '17', '18']

# Number of HRUs in each region
hrus_by_region = [2462, 4827, 9899, 5936, 7182, 2303, 8205, 4449, 
                  1717, 8603, 10299, 7373, 7815, 1958, 3879, 3441,
                  2664, 11102, 5837]


def load_MWBM_streamflow(filename, st_date, en_date): 
    # Load MWBM streamflow error bounds
    df = pd.read_csv(filename, sep=' ', skipinitialspace=True,
                     date_parser=prms_objfcn.dparse,
                     parse_dates={'thedate': [0, 1]}, index_col=['thedate'])
    df = df.resample('M', how='mean')
    return df[st_date:en_date]


def pull_by_hru(src_dir, dst_dir, st_date, en_date, region, param_file):
    # For a given region pull MWBM streamflow for each HRU and write it to the dst_dir
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

    # Read in the hru_area from the input parameter file
    print "Reading HRU areas from %s" % param_file
    param = prms.parameters(param_file)

    # hru_area is in units of acres
    hru_area = param.get_var('hru_area')['values']
    print 'Number of HRU area values:', len(hru_area)

    print "Loading MWBM streamflow:"
    print "\tMinimum bounds from %s/roHRUminERR_r%s" % (src_dir, region)
    ds1 = load_MWBM_streamflow('%s/roHRUminERR_r%s' % (src_dir, region), st_date, en_date)

    print "\tMaximum bounds from %s/roHRUmaxERR_r%s" % (src_dir, region)
    ds2 = load_MWBM_streamflow('%s/roHRUmaxERR_r%s' % (src_dir, region), st_date, en_date)

    print "Writing out HRUs:"
    for hh in xrange(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        # Create and convert minimum values for HRU
        ds1_hru = pd.DataFrame(ds1.ix[:,hh])
        ds1_hru.rename(columns={ds1_hru.columns[0]: 'min'}, inplace=True)

        # Convert from mm/month to cfs (see Evernote for units conversion notes)
        # hru_area is converted from acres to square kilometers
        ds1_hru['min'] = ds1_hru['min'] * hru_area[hh] * 0.0016540916
        ds1_hru['min'] /= ds1_hru.index.day
        #ds1_hru['min'] = ds1_hru * (hru_area[hh] * 0.00404686) / 2.4465755462

        # Create and convert maximum values for HRU
        ds2_hru = pd.DataFrame(ds2.ix[:,hh])
        ds2_hru.rename(columns={ds2_hru.columns[0]: 'max'}, inplace=True)

        # Convert from mm/month to cfs (see Evernote for units conversion notes)
        ds2_hru['max'] = ds2_hru['max'] * hru_area[hh] * 0.0016540916
        ds2_hru['max'] /= ds2_hru.index.day
        #ds2_hru['max'] = ds2_hru * (hru_area[hh] * 0.00404686) / 2.4465755462

        # Join min and max value datasets together
        ds_join = ds1_hru.join(ds2_hru, how='outer')
        #ds_join = ds_join[st_date:en_date]
        ds_join['Year'] = ds_join.index.year
        ds_join['Month'] = ds_join.index.month
        ds_join.reset_index(inplace=True)
        ds_join.drop(['thedate'], axis=1, inplace=True)

        # Wrie out the dataset
        outfile = '%s/r%s_%06d/MWBMerror' % (dst_dir, region, hh)
        ds_join.to_csv(outfile, sep=' ', float_format='%0.5f', columns=['Year', 'Month', 'min', 'max'], 
                       header=False, index=False)
    print ''


def pull_by_hru_GCPO(src_dir, dst_dir, st_date, en_date, region, param_file):
    # For a given region pull MWBM streamflow for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir        Location of the AET datasets
    # dstdir        Top-level location to write HRUs
    # st_date       Start date for output dataset
    # en_date       End date for output datasdet
    # region        The region to pull HRUs out of
    # param_file    Input parameter file to obtain hru_area from

    # Override the region information for the GCPO
    regions = ['GCPO']
    hrus_by_region = [20251]

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # Read in the hru_area from the input parameter file
    print "Reading HRU areas from %s" % param_file
    param = prms.parameters(param_file)

    # hru_area is in units of acres
    hru_area = param.get_var('hru_area')['values']
    print 'Number of HRU area values:', len(hru_area)

    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    print "Loading MWBM streamflow:"
    print "\tMedian runoff error from %s/MWBM_RO_HRU50ERR_GCPO_1980-2010" % src_dir
    ro_err_mwbm = pd.read_csv('%s/MWBM_RO_HRU50ERR_GCPO_1980-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser, index_col='thedate')
    # ds1 = load_MWBM_streamflow('%s/roHRUminERR_r%s' % (src_dir, region), st_date, en_date)

    print "\tRunoff from %s/MWBM_RO_HRU_GCPO_1980-2010" % src_dir
    ro_mwbm = pd.read_csv('%s/MWBM_RO_HRU_GCPO_1980-2010' % src_dir , sep=' ', parse_dates=True, date_parser=parser, index_col='thedate')
    # ds2 = load_MWBM_streamflow('%s/roHRUmaxERR_r%s' % (src_dir, region), st_date, en_date)

    print "Writing out HRUs:"
    for hh in xrange(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        # Create and convert actual runoff values for HRU
        ro_hru = pd.DataFrame(ro_mwbm.iloc[:,hh])
        ro_hru.rename(columns={ro_hru.columns[0]: 'runoff'}, inplace=True)

        # Create and convert median runoff error for HRU
        ro_err_hru = pd.DataFrame(ro_err_mwbm.iloc[:,hh])
        ro_err_hru.rename(columns={ro_err_hru.columns[0]: 'error'}, inplace=True)

        # Minimum error value
        ro_err_hru['min'] = ro_hru['runoff'] - ro_err_hru['error'].abs()

        # Sometimes the min value is negative, set those to zero
        ro_err_hru['min'] = np.where(ro_err_hru['min']<0.0, 0.0, ro_err_hru['min'])

        # Maximum error value
        ro_err_hru['max'] = ro_hru['runoff'] + ro_err_hru['error'].abs()

        # Convert from mm/month to cfs (see Evernote for units conversion notes)
        # hru_area is converted from acres to square kilometers
        ro_err_hru['min'] = ro_err_hru['min'] * hru_area[hh] * 0.0016540916
        ro_err_hru['min'] /= ro_err_hru.index.day

        ro_err_hru['max'] = ro_err_hru['max'] * hru_area[hh] * 0.0016540916
        ro_err_hru['max'] /= ro_err_hru.index.day

        ro_err_hru['Year'] = ro_err_hru.index.year
        ro_err_hru['Month'] = ro_err_hru.index.month
        ro_err_hru.reset_index(inplace=True)
        ro_err_hru.drop(['thedate', 'error'], axis=1, inplace=True)

        # Write out the dataset
        outfile = '%s/r%s_%06d/MWBMerror' % (dst_dir, region, hh)
        ro_err_hru.to_csv(outfile, sep=' ', float_format='%.3f', columns=['Year', 'Month', 'min', 'max'],
                          header=False, index=False)
    print ''


def pull_RO_by_hru_GCPO(src_dir, dst_dir, st_date, en_date, region, param_file):
    # For a given region pull MWBM streamflow for each HRU and write it to the dst_dir
    #
    # Parameters:
    # srcdir        Location of the AET datasets
    # dstdir        Top-level location to write HRUs
    # st_date       Start date for output dataset
    # en_date       End date for output datasdet
    # region        The region to pull HRUs out of
    # param_file    Input parameter file to obtain hru_area from

    # Override the global region information for the GCPO
    regions = ['GCPO']
    hrus_by_region = [20251]

    if not isinstance(st_date, datetime.datetime):
        date_split = st_date.split('-')
        st_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    if not isinstance(en_date, datetime.datetime):
        date_split = en_date.split('-')
        en_date = datetime.datetime(date_split[0], date_split[1], date_split[2])

    # Read in the hru_area from the input parameter file
    print "Reading HRU areas from %s" % param_file
    param = prms.parameters(param_file)

    # hru_area is in units of acres
    hru_area = param.get_var('hru_area')['values']
    print 'Number of HRU area values:', len(hru_area)

    # Parser for the date information
    parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    print "\tRunoff from %s/MWBM_RO_HRU_GCPO_1980-2010" % src_dir
    ro_mwbm = pd.read_csv('%s/MWBM_RO_HRU_GCPO_1980-2010' % src_dir, sep=' ', parse_dates=True, date_parser=parser,
                          index_col='thedate')
    # ds2 = load_MWBM_streamflow('%s/roHRUmaxERR_r%s' % (src_dir, region), st_date, en_date)

    print "Writing out HRUs:"
    for hh in xrange(hrus_by_region[regions.index(region)]):
        sys.stdout.write('\r\t%06d ' % hh)
        sys.stdout.flush()

        # Create and convert actual runoff values for HRU
        ro_hru = pd.DataFrame(ro_mwbm.iloc[:, hh])
        ro_hru.rename(columns={ro_hru.columns[0]: 'runoff'}, inplace=True)

        # Convert from mm/month to cfs (see Evernote for units conversion notes)
        # hru_area is converted from acres to square kilometers
        ro_hru['runoff'] = ro_hru['runoff'] * hru_area[hh] * 0.0016540916
        ro_hru['runoff'] /= ro_hru.index.day

        ro_hru['Year'] = ro_hru.index.year
        ro_hru['Month'] = ro_hru.index.month
        ro_hru.reset_index(inplace=True)
        ro_hru.drop(['thedate'], axis=1, inplace=True)

        # Write out the dataset
        outfile = '%s/r%s_%06d/MWBM' % (dst_dir, region, hh)
        ro_hru.to_csv(outfile, sep=' ', float_format='%.3f', columns=['Year', 'Month', 'runoff'], header=False, index=False)
    print ''


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Split MWBM model output into individual HRUs')
    parser.add_argument('-b', '--basedir', help='Base directory for regions', required=True)
    parser.add_argument('-s', '--srcdir', help='MWBM directory', required=True)
    parser.add_argument('-p', '--paramfile', help='Name of the full input parameter file', required=True)
    parser.add_argument('-r', '--region', help='Region to process', required=True)
    parser.add_argument('--range', help='Create error range files', action='store_true')

    args = parser.parse_args()

    selected_region = args.region
    base_dir = args.basedir
    src_dir = args.srcdir
    dst_dir = '%s/r%s_byHRU' % (base_dir, args.region)
    param_file = '%s/r%s/%s' % (base_dir, args.region, args.paramfile)

    # selected_region = 'GCPO'
    # src_dir = '/media/scratch/PRMS/datasets/MWBMerr'
    # dst_dir = '/media/scratch/PRMS/regions/r%s_byHRU' % selected_region
    # param_file = '/media/scratch/PRMS/regions/r%s/daymet.params' % selected_region

    st = datetime.datetime(1980, 1, 1)
    en = datetime.datetime(2010, 12, 31)

    if selected_region == 'GCPO':
        if args.range:
            pull_by_hru_GCPO(src_dir, dst_dir, st, en, selected_region, param_file)
        else:
            pull_RO_by_hru_GCPO(src_dir, dst_dir, st, en, selected_region, param_file)
    else:
        pull_by_hru(src_dir, dst_dir, st, en, selected_region, param_file)

if __name__ == '__main__':
    main()

