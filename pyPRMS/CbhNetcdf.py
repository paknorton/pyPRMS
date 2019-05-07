from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

import pandas as pd
import netCDF4 as nc
import xarray as xr

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class CbhNetcdf(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-30
    # Description: Class for working with individual cbh files

    # 2016-12-20 PAN:
    # As written this works with CBH files that were created with
    # java class gov.usgs.mows.GCMtoPRMS.GDPtoCBH
    # This program creates the CBH files in the format needed by PRMS
    # and also verifies the correctness of the data including:
    #    tmax is never less than tmin
    #    prcp is never negative
    #    any missing data/missing date is filled with (?? avg of bracketing dates??)
    #
    # I think it would be better if this code worked with the original GDP files and
    # took care of those corrections itself. This would provide a more seamless workflow
    # from GDP to PRMS. At this point I'm not taking this on though -- for a future revision.

    def __init__(self, src_path=None, st_date=None, en_date=None, nhm_hrus=None):
        self.__src_path = src_path
        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus
        self.__date_range = None
        self.__dataset = None
        self.__final_outorder = None

        self.read_netcdf()

    def read_netcdf(self):
        """Read CBH files stored in netCDF format"""

        if self.__nhm_hrus:
            # print('\t\tOpen dataset')
            # self.__dataset = xr.open_mfdataset(self.__src_path)
            self.__dataset = xr.open_mfdataset(self.__src_path, chunks={'hru': 1040})
        else:
            print('ERROR: write the code for all HRUs')
            exit()

    def get_var(self, var):
        if self.__stdate is not None and self.__endate is not None:
            data = self.__dataset[var].loc[self.__stdate:self.__endate, self.__nhm_hrus].to_pandas()
        else:
            data = self.__dataset[var].loc[:, self.__nhm_hrus].to_pandas()

        return data

    def write_ascii(self, pathname=None, fileprefix=None, vars=None):
        # For out_order the first six columns contain the time information and
        # are always output for the cbh files
        out_order = [kk for kk in self.__nhm_hrus]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        if vars is None:
            var_list = self.__dataset.data_vars
        elif isinstance(list, vars):
            var_list = vars

        for cvar in var_list:
            data = self.get_var(var=cvar)

            # Add time information as columns
            data['year'] = data.index.year
            data['month'] = data.index.month
            data['day'] = data.index.day
            data['hour'] = 0
            data['minute'] = 0
            data['second'] = 0

            # Output ASCII CBH files
            if fileprefix is None:
                outfile = '{}.cbh'.format(cvar)
            else:
                outfile = '{}_{}.cbh'.format(fileprefix, cvar)

            if pathname is not None:
                outfile = '{}/{}'.format(pathname, outfile)

            out_cbh = open(outfile, 'w')
            out_cbh.write('Written by Bandit\n')
            out_cbh.write('{} {}\n'.format(cvar, len(self.__nhm_hrus)))
            out_cbh.write('########################################\n')
            # data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.3f',
            data.to_csv(out_cbh, columns=out_order, na_rep='-999',
                        sep=' ', index=False, header=False, encoding=None, chunksize=50)
            out_cbh.close()

    def write_netcdf(self, filename=None, vars=None):
        """Write CBH to netcdf format file"""

        # NetCDF-related variables
        var_desc = {'tmax': 'Maximum Temperature', 'tmin': 'Minimum temperature', 'prcp': 'Precipitation'}
        var_units = {'tmax': 'C', 'tmin': 'C', 'prcp': 'inches'}

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)
        # nco.createDimension('hru', len(self.__dataset['hru'].loc[self.__nhm_hrus]))
        nco.createDimension('hru', len(self.__nhm_hrus))
        nco.createDimension('time', None)

        timeo = nco.createVariable('time', 'f4', ('time'))
        timeo.calendar = 'standard'
        # timeo.bounds = 'time_bnds'
        timeo.units = 'days since 1980-01-01 00:00:00'

        hruo = nco.createVariable('hru', 'i4', ('hru'))
        hruo.long_name = 'Hydrologic Response Unit ID (HRU)'

        if vars is None:
            var_list = self.__dataset.data_vars
        elif isinstance(list, vars):
            var_list = vars

        for cvar in var_list:
            varo = nco.createVariable(cvar, 'f4', ('time', 'hru'), fill_value=nc.default_fillvals['f4'], zlib=True)
            varo.long_name = var_desc[cvar]
            varo.units = var_units[cvar]

        nco.setncattr('Description', 'Climate by HRU')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the HRU ids
        hruo[:] = self.__dataset['hru'].loc[self.__nhm_hrus].values
        hruo[:] = self.__nhm_hrus
        timeo[:] = nc.date2num(pd.to_datetime(self.__dataset['time'].loc[self.__stdate:self.__endate].values).tolist(),
                               units='days since 1980-01-01 00:00:00',
                               calendar='standard')

        for cvar in var_list:
            data = self.get_var(var=cvar)

            # Write the CBH values
            nco.variables[cvar][:, :] = data.values

        nco.close()

