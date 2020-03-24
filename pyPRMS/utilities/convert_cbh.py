#!/usr/bin/env python

import argparse
import datetime
import netCDF4
import os

from pyPRMS.CbhAscii import CbhAscii
from pyPRMS.ParamDbRegion import ParamDbRegion

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Convert parameter files to different formats')
    parser.add_argument('-s', '--start_date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'),
                        help='Starting date in the format YYYY-MM-DD')
    parser.add_argument('-e', '--end_date', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'),
                        help='Ending date in the format YYYY-MM-DD')
    parser.add_argument('-P', '--paramdb_dir', help='Location of parameter database')
    parser.add_argument('-C', '--cbh_path', help='Location of CBH files')
    parser.add_argument('--dst', help='Destination file or directory')
    parser.add_argument('--prefix', help='Filename prefix')
    parser.add_argument('-v', '--var', help='Variable to process')

    args = parser.parse_args()

    st_date = args.start_date
    en_date = args.end_date
    # st_date = datetime.datetime(1980, 1, 1)
    # en_date = datetime.datetime(2016, 12, 31)

    print('Reading NHM paramDb')
    nhm_pdb = ParamDbRegion(paramdb_dir=args.paramdb_dir, verbose=True, verify=True)
    hru_nhm_to_local = nhm_pdb.hru_nhm_to_local
    hru_nhm_to_region = nhm_pdb.hru_nhm_to_region

    # Check and read the source
    print('Reading source CBH file(s)')
    if not os.path.isdir(args.cbh_path):
        print('ERROR: cbh_path should be a directory where CBH files are located')
        exit(-1)

    # check for / create output directory
    try:
        os.makedirs(args.dst)
        print('Creating directory for output: {}'.format(args.dst))
    except OSError:
        print("\tUsing existing directory for output: {}".format(args.dst))

    cbh_hdl = CbhAscii(src_path=args.cbh_path, indices=hru_nhm_to_local,
                       nhm_hrus=list(hru_nhm_to_local.keys()),
                       mapping=hru_nhm_to_region, st_date=st_date, en_date=en_date)
    data = cbh_hdl.read_cbh_multifile(var=args.var)

    for cyear in range(st_date.year, en_date.year + 1):
        print(cyear)
        c_start = datetime.datetime(cyear, 1, 1)
        c_end = datetime.datetime(cyear, 12, 31)

        # NetCDF-related variables
        var_desc = {'tmax': 'Maximum Temperature', 'tmin': 'Minimum temperature', 'prcp': 'Precipitation'}
        var_units = {'tmax': 'C', 'tmin': 'C', 'prcp': 'inches'}

        # Create a netCDF file for the CBH data
        nco = netCDF4.Dataset('{}/{}_{}_{}-{}.nc'.format(args.dst, args.prefix, args.var,
                                                         c_start.strftime('%Y%m%d'),
                                                         c_end.strftime('%Y%m%d')), 'w', clobber=True)
        nco.createDimension('hru', len(list(hru_nhm_to_local.keys())))
        nco.createDimension('time', None)

        timeo = nco.createVariable('time', 'f4', ('time'))
        timeo.calendar = 'standard'
        # timeo.bounds = 'time_bnds'
        # timeo.units = 'days since 1980-01-01 00:00:00'
        timeo.units = 'days since {}-{:02d}-{:02d} 00:00:00'.format(st_date.year, st_date.month, 1)

        hruo = nco.createVariable('hru', 'i4', ('hru'))
        hruo.long_name = 'Hydrologic Response Unit ID (HRU)'

        varo = nco.createVariable(args.var, 'f4', ('time', 'hru'), fill_value=netCDF4.default_fillvals['f4'],
                                  zlib=True, complevel=1, chunksizes=[31, 260])
        varo.long_name = var_desc[args.var]
        varo.units = var_units[args.var]

        nco.setncattr('Description', 'Climate by HRU')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the HRU ids
        hruo[:] = list(hru_nhm_to_local.keys())

        curr_calendar = timeo.calendar
        curr_units = timeo.units

        timeo[:] = netCDF4.date2num(data[c_start:c_end].index.tolist(), units=curr_units,
                                    calendar=curr_calendar)

        # Write the CBH values
        nco.variables[args.var][:, :] = data[c_start:c_end].values

    nco.close()


if __name__ == '__main__':
    main()

