#!/usr/bin/env python

# Process streamflow from the Monthly Water Balance Model (MWBM) into individual
# HRUs for a given region for use by PRMS model calibration by HRU.

import prms_lib as prms
import pandas as pd
import numpy as np
import datetime
import sys
# The runoff error bounds from the Monthly Water Balance Model (MWBM) are located on yeti at /cxfs/projects/usgs/water/mows/MWBMRUNSwGRPparams/MWBMerr. The runoff is broken out by region with a single file for each min and max set of values. Runoff values are in units of mm/day. To convert from mm/day to cubic feet per second the following conversion is used.
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

        # Convert from mm/day to cfs (see Evernote for units conversion notes)
        # hru_area is converted from acres to square kilometers
        ds1_hru['min'] = ds1_hru * (hru_area[hh] * 0.00404686) / 2.4465755462

        # Create and convert maximum values for HRU
        ds2_hru = pd.DataFrame(ds2.ix[:,hh])
        ds2_hru.rename(columns={ds2_hru.columns[0]: 'max'}, inplace=True)

        # Convert from mm/day to cfs (see Evernote for units conversion notes)
        ds2_hru['max'] = ds2_hru * (hru_area[hh] * 0.00404686) / 2.4465755462

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



def main():
    selected_region = '10U'
    src_dir = '/media/scratch/PRMS/datasets/MWBMerr'
    dst_dir = '/media/scratch/PRMS/regions/r%s_byHRU' % selected_region

    param_file = '/media/scratch/PRMS/regions/r10U/input/params/daymet.params'

    st = datetime.datetime(1980,1,1)
    en = datetime.datetime(2010,12,31)

    pull_by_hru(src_dir, dst_dir, st, en, selected_region, param_file)

if __name__ == '__main__':
    main()

