#!/usr/bin/env python2.7

import datetime
import pandas as pd
import numpy as np
import prms_lib as prms
from prms_objfcn import dparse


regions = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
           '10U', '10L', '11', '12', '13', '14', '15', '16', '17', '18']
#regions = ['01']

#event_types = ['liquid', 'snow']
#event_type = 'liquid' # one of snow, liquid
#region = '01'

base_daymet_dir = '/media/scratch/PRMS/datasets/daymet'
base_snodas_dir = '/media/scratch/PRMS/datasets/snodasPrecip'
prms_region_dir = '/media/scratch/PRMS/regions'

rain_max = 38.    # based on 90% probability of rain
rain_min = 34.7   # based on 50% 
snow_max = 34.25  # based on 50% probability of snow
snow_min = 28.2   # based on 95%

st = datetime.datetime(2004,1,1)
en = datetime.datetime(2010,12,31)


def main():
    for rr in regions:
        print 'Region: r%s' % rr

        # Open the master input parameter file for this region
        prms_region_file = '%s/r%s/daymet.control.param' % (prms_region_dir, rr)
        prms_region_outfile = '%s/r%s/daymet.params.update.tmax_allsnow' % (prms_region_dir, rr)

        # Open input parameter file
        params = prms.parameters(prms_region_file)

        tmaxfile = '%s/r%s/daymet_1980_2011_tmax.cbh' % (base_daymet_dir, rr)
        prcpfile = '%s/r%s/daymet_1980_2011_prcp.cbh' % (base_daymet_dir, rr)

        # Read the daymet tmax and prcp cbh file
        # Read the tmax CBH file and restrict to the given date range
        df_tmax = prms.read_cbh(tmaxfile)
        df_tmax = df_tmax[st:en]

        # Read the prcp CBH file and restrict to the given date range
        df_prcp = prms.read_cbh(prcpfile)
        df_prcp = df_prcp[st:en]

        # Create mask from precip values
        # We'll use this to minimimize alignment issues between the daymet and
        # snodas datasets
        df_prcp[df_prcp == 0.0] = np.nan
        df_prcp[df_prcp > 0.0] = 1


        # Load the SNODAS precip masks
        rainmaskfile = '%s/r%s/liquid_only_SNODASPRCP_Daily_r%s_byHRU_2004-01-01_2014-12-31.csv' % (base_snodas_dir, rr, rr)
        snowmaskfile = '%s/r%s/snow_only_SNODASPRCP_Daily_r%s_byHRU_2004-01-01_2014-12-31.csv' % (base_snodas_dir, rr, rr)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load the rain (or liquid) precip mask
        print '\tCreate rain mask'
        rain_mask =  pd.read_csv(rainmaskfile, na_values=['MAXIMUM(none)', 255.0], header=0, skiprows=[0,2])

        rain_mask.rename(columns={rain_mask.columns[0]:'thedate'}, inplace=True)
        rain_mask['thedate'] = pd.to_datetime(rain_mask['thedate'])
        rain_mask.set_index('thedate', inplace=True)
        rain_mask = rain_mask[st:en]

        # Fill in any missing days with NAN
        rain_mask = rain_mask.resample('D', how='max')
        rain_mask[rain_mask > 0.0] = 1

        # Use numpy (np) to multiply df_tmax and df_mask.
        # For some reason pandas multiply is horribly broken
        rain_tmax_m1 = pd.np.multiply(df_tmax,rain_mask)
        rain_tmax_m1[rain_tmax_m1 == 0.0] = np.nan

        # Create secondary mask
        rain_tmax_m2 = pd.np.multiply(df_tmax, df_prcp)

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

        rain_tmax_final.index.name = 'HRU'


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load the snow precip mask
        print '\tCreate snow mask'
        snow_mask =  pd.read_csv(snowmaskfile, na_values=['MAXIMUM(none)', 255.0], header=0, skiprows=[0,2])

        snow_mask.rename(columns={snow_mask.columns[0]:'thedate'}, inplace=True)
        snow_mask['thedate'] = pd.to_datetime(snow_mask['thedate'])
        snow_mask.set_index('thedate', inplace=True)
        snow_mask = snow_mask[st:en]

        # Fill in any missing days with NAN
        snow_mask = snow_mask.resample('D', how='max')
        snow_mask[snow_mask > 0.0] = 1

        # Use numpy (np) to multiply df_tmax and df_mask.
        # For some reason pandas multiply is horribly broken
        snow_tmax_m1 = pd.np.multiply(df_tmax,snow_mask)
        snow_tmax_m1[snow_tmax_m1 == 0.0] = np.nan

        # Create secondary mask
        snow_tmax_m2 = pd.np.multiply(df_tmax, df_prcp)

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

        snow_tmax_final.index.name = 'HRU'

        # Compute the tmax_allrain_offset value
        rain_tmax_offset = pd.np.subtract(rain_tmax_final, snow_tmax_final)

        print '\tUpdate parameters and write out difference file'
        # Update tmax_allsnow and tmax_allrain_offset
        params.replace_values('tmax_allsnow', snow_tmax_final.T.values)
        params.replace_values('tmax_allrain_offset', rain_tmax_offset.T.values)

        # Write out new input parameter file
        #params.write_param_file(prms_region_outfile)
        params.write_select_param_file(prms_region_outfile, ['tmax_allsnow', 'tmax_allrain_offset'])


if __name__ == '__main__':
    main()
