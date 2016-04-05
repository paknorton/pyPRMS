#!/usr/bin/env python2.7

import datetime
import pandas as pd
import numpy as np
import prms_lib as prms
# from prms_objfcn import dparse

region_list = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08',
           'r09', 'r10L', 'r10U', 'r11', 'r12', 'r13', 'r14', 'r15',
           'r16', 'r17', 'r18']

hrus_by_region = [2462, 4827, 9899, 5936, 7182, 2303, 8205, 4449,
                  1717, 8603, 10299, 7373, 7815, 1958, 3879, 3441,
                  2664, 11102, 5837]

rain_max = 38.    # based on 90% probability of rain
rain_min = 34.7   # based on 50% 
snow_max = 34.25  # based on 50% probability of snow
snow_min = 28.2   # based on 95%

# st = datetime.datetime(2004,1,1)
# en = datetime.datetime(2010,12,31)

def create_rain_tmax(rainmask, tmax, prcp):
    """Compute tmax_allrain from a rain mask and observed tmax and prcp from cbh files"""
    # Fill in any missing days with NAN
    rainmask = rainmask.resample('D', how='max')
    rainmask[rainmask > 0.0] = 1

    # Use numpy (np) to multiply df_tmax and df_mask.
    # For some reason pandas multiply is horribly broken
    rain_tmax_m1 = pd.np.multiply(tmax,rainmask)
    rain_tmax_m1[rain_tmax_m1 == 0.0] = np.nan

    # Create secondary mask
    rain_tmax_m2 = pd.np.multiply(tmax, prcp)

    # Mask out temperatures values outside the range we need
    rain_tmax_m1[rain_tmax_m1 > rain_max] = np.nan
    rain_tmax_m1[rain_tmax_m1 < rain_min] = np.nan
    rain_tmax_m2[rain_tmax_m2 > rain_max] = np.nan
    rain_tmax_m2[rain_tmax_m2 < rain_min] = np.nan

    # Resample to monthly mean
    rain_tmax_m1_mon = rain_tmax_m1.resample('M', how='mean')

    # compute mean monthly tmax for each hru
    rain_tmax_m1_mnmon = rain_tmax_m1_mon.groupby(rain_tmax_m1_mon.index.month).mean()

    # Do the same for the secondary masked set
    rain_tmax_m2_mon = rain_tmax_m2.resample('M', how='mean')
    rain_tmax_m2_mnmon = rain_tmax_m2_mon.groupby(rain_tmax_m2_mon.index.month).mean()

    # Where rain_m1_mnmon is NaN replace with value from rain_m2_mnmon if it exists
    # otherwise replace with default value

    # First replace any Nan entries with a valid entry from rain_tmax_m2_mnmon
    # Then replace any remaining NaN entries by padding them with the last valid value
    rain_tmax_final = rain_tmax_m1_mnmon.fillna(value=rain_tmax_m2_mnmon)
    rain_tmax_final.fillna(value=38., inplace=True)

    rain_tmax_final.index.name = 'Month'
    return rain_tmax_final


def create_snow_tmax(snowmask, tmax, prcp):
    """Compute tmax_allsnow from a snow mask and observed tmax and prcp from cbh files"""
    # Fill in any missing days with NAN
    snowmask = snowmask.resample('D', how='max')
    snowmask[snowmask > 0.0] = 1

    # Use numpy (np) to multiply df_tmax and df_mask.
    # For some reason pandas multiply is horribly broken
    snow_tmax_m1 = pd.np.multiply(tmax,snowmask)
    snow_tmax_m1[snow_tmax_m1 == 0.0] = np.nan

    # Create secondary mask
    snow_tmax_m2 = pd.np.multiply(tmax, prcp)

    # Mask out temperatures values outside the range we need
    snow_tmax_m1[snow_tmax_m1 > snow_max] = np.nan
    snow_tmax_m1[snow_tmax_m1 < snow_min] = np.nan

    snow_tmax_m2[snow_tmax_m2 > snow_max] = np.nan
    snow_tmax_m2[snow_tmax_m2 < snow_min] = np.nan

    # Resample to monthly mean
    snow_tmax_m1_mon = snow_tmax_m1.resample('M', how='mean')

    # compute mean monthly tmax for each hru
    snow_tmax_m1_mnmon = snow_tmax_m1_mon.groupby(snow_tmax_m1_mon.index.month).mean()

    # Do the same for the secondary masked set
    snow_tmax_m2_mon = snow_tmax_m2.resample('M', how='mean')
    snow_tmax_m2_mnmon = snow_tmax_m2_mon.groupby(snow_tmax_m2_mon.index.month).mean()

    # Where snow_m1_mnmon is NaN replace with value from snow_m2_mnmon if it exists
    # otherwise replace with default value

    # First replace any Nan entries with a valid entry from snow_tmax_m2_mnmon
    # Then replace any remaining NaN entries by padding them with the last valid value
    snow_tmax_final = snow_tmax_m1_mnmon.fillna(value=snow_tmax_m2_mnmon)
    snow_tmax_final.fillna(value=32., inplace=True)

    snow_tmax_final.index.name = 'Month'
    return snow_tmax_final


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create tmax_allsnow and tmax_allrain_offset for PRMS')
    parser.add_argument('-c', '--cbhdir', help='Base directory for cbh files by region', required=True)
    parser.add_argument('-d', '--daterange', help='Starting and ending date (YYYY-MM-DD YYYY-MM-DD)',
                        nargs='*', metavar=('startDate','endDate'), required=True)
    parser.add_argument('-o', '--outputfile', help='Output parameter filename')
    parser.add_argument('-p', '--paramfile', help='Input parameter filename in region(s)')
    parser.add_argument('-P', '--prcpfile', help='Name of prcp cbh file in region(s)', required=True)
    parser.add_argument('-r', '--regiondir', help='Base directory for PRMS models by region', required=True)
    parser.add_argument('-R', '--regions', help='Region(s) to process', nargs='*', required=True)
    parser.add_argument('-s', '--snodasdir', help='Base directory for snodas files by region', required=True)
    parser.add_argument('-T', '--tmaxfile', help='Name of tmax cbh file in region(s)', required=True)

    parser.add_argument('--makeparam', help='Make parameter diff files', action='store_true')
    parser.add_argument('--makemerge', help='Create merged file from selected regions', action='store_true')

    args = parser.parse_args()

    # Verify and set the date range
    if len(args.daterange) < 2:
        print 'ERROR: A start date or start and ending date must be specified'
        exit()
    else:
        st = prms.to_datetime(args.daterange[0])
        en = prms.to_datetime(args.daterange[1])
        print 'Date Range: %s to %s' % (args.daterange[0], args.daterange[1])

    if len(args.regions) == 0:
        print 'ERROR: Must supply at least one region to process'
        exit()
    else:
        regions = args.regions

    prms_region_dir = args.regiondir
    base_cbh_dir = args.cbhdir
    base_snodas_dir = args.snodasdir

    for ridx, rr in enumerate(regions):
        print 'Region: %s' % rr

        # Set various files
        rainmaskfile = '%s/%s/liquid_only_SNODASPRCP_Daily_%s_byHRU_2004-01-01_2014-12-31.csv' % \
                       (base_snodas_dir, rr, rr)
        snowmaskfile = '%s/%s/snow_only_SNODASPRCP_Daily_%s_byHRU_2004-01-01_2014-12-31.csv' % \
                       (base_snodas_dir, rr, rr)

        tmaxfile = '%s/%s/%s' % (base_cbh_dir, rr, args.tmaxfile)
        prcpfile = '%s/%s/%s' % (base_cbh_dir, rr, args.prcpfile)

        if args.makeparam:
            prms_region_file = '%s/%s/%s' % (prms_region_dir, rr, args.paramfile)
            prms_region_outfile = '%s/%s/%s' % (prms_region_dir, rr, args.outputfile)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Open input parameter file for current region
            params = prms.parameters(prms_region_file)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read the daymet tmax and prcp cbh file
        # Read the tmax CBH file and restrict to the given date range
        print '\tReading tmax CBH file'
        df_tmax = prms.read_cbh(tmaxfile)
        df_tmax = df_tmax[st:en]

        # Read the prcp CBH file and restrict to the given date range
        print '\tReading prcp CBH file'
        df_prcp = prms.read_cbh(prcpfile)
        df_prcp = df_prcp[st:en]

        # Create mask from precip values
        # We'll use this to minimimize alignment issues between the daymet and
        # snodas datasets
        df_prcp[df_prcp == 0.0] = np.nan
        df_prcp[df_prcp > 0.0] = 1

        # Load the SNODAS precip masks
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load the rain (or liquid) precip mask
        print '\tCreate rain mask'
        rain_mask = prms.read_gdp(rainmaskfile, missing_val=255.)
        rain_mask = rain_mask[st:en]
        rain_tmax_final = create_rain_tmax(rain_mask, df_tmax, df_prcp)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load the snow precip mask
        print '\tCreate snow mask'
        snow_mask = prms.read_gdp(snowmaskfile, missing_val=255.)
        snow_mask = snow_mask[st:en]
        snow_tmax_final = create_snow_tmax(snow_mask, df_tmax, df_prcp)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the tmax_allrain_offset value
        rain_tmax_offset = pd.np.subtract(rain_tmax_final, snow_tmax_final)

        if args.makeparam:
            if not (params.var_exists('tmax_allsnow') or params.var_exists('tmax_allrain_offset')):
                print "ERROR: tmax_allsnow and tmax_allrain_offset must exist in the parameter file"
                exit(1)

            # Update tmax_allsnow and tmax_allrain_offset in the input parameter object
            print '\tUpdate parameters and write out difference file'
            params.replace_values('tmax_allsnow', snow_tmax_final.values.flatten())
            params.replace_values('tmax_allrain_offset', rain_tmax_offset.values.flatten())
            # params.replace_values('tmax_allsnow', snow_tmax_final.values)
            # params.replace_values('tmax_allrain_offset', rain_tmax_offset.values)

            # Write out new input parameter file
            params.write_select_param_file(prms_region_outfile, ['tmax_allsnow', 'tmax_allrain_offset'])

        if args.makemerge:
            # Rename columns to their national id number
            # start_idx is zero-based and is convert
            start_idx = sum(hrus_by_region[0:region_list.index(rr)])
            snow_tmax_final.rename(columns=lambda x: snow_tmax_final.columns.get_loc(x)+start_idx+1, inplace=True)
            rain_tmax_offset.rename(columns=lambda x: rain_tmax_offset.columns.get_loc(x)+start_idx+1, inplace=True)

            if ridx == 0:
                tmax_allsnow = snow_tmax_final
                tmax_allrain_offset = rain_tmax_offset
            else:
                tmax_allsnow = pd.concat([tmax_allsnow, snow_tmax_final], axis=1)
                tmax_allrain_offset = pd.concat([tmax_allrain_offset, rain_tmax_offset], axis=1)

    # If requested, write out the merge files
    if args.makemerge:
        tmax_allsnow.to_csv('%s/tmax_allsnow_merged.csv' % base_snodas_dir, index=True, float_format='%7.4f')
        tmax_allrain_offset.to_csv('%s/tmax_allrain_offset_merged.csv' % base_snodas_dir, index=True, float_format='%7.4f')

if __name__ == '__main__':
    main()
