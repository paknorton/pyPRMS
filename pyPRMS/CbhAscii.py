import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from collections import OrderedDict
from typing import Union, OrderedDict as OrderedDictType

from pyPRMS.prms_helpers import dparse
from pyPRMS.constants import REGIONS

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class CbhAscii(object):

    """Class for handling classic climate-by-hru (CBH) files.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04

    # This class assumes it is dealing with regional cbh files (not a CONUS-level NHM file)
    # TODO: As currently written type of data (e.g. tmax, tmin, prcp) is ignored.
    # TODO: Verify that given data type size matches number of columns

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

    def __init__(self, src_path=None, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None):
        """Create CbhAscii object.
        """

        self.__src_path = src_path

        # self.__indices = [str(kk) for kk in indices]
        self.__indices = indices    # OrdereDict: nhm_ids -> local_ids

        self.__data = None
        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus
        self.__mapping = mapping
        self.__date_range = None
        self.__dataframe = None
        self.__dataset = None
        self.__final_outorder = None

    def read_cbh(self):
        """Reads an entire CBH file.
        """

        # incl_cols = list(self.__indices.values())
        # for xx in CBH_INDEX_COLS[:-1]:
        #     incl_cols.insert(0, xx)

        incl_cols = list(CBH_INDEX_COLS)

        for xx in self.__indices.values():
            incl_cols.append(xx+5)  # include an offset for the having datetime info
        # print(incl_cols)

        # Columns 0-5 always represent date/time information
        self.__data = pd.read_csv(self.__src_path, sep=' ', skipinitialspace=True, usecols=incl_cols,
                                  skiprows=3, engine='c', memory_map=True,
                                  date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                  index_col='time', header=None, na_values=[-99.0, -999.0])

        if self.__stdate is not None and self.__endate is not None:
            self.__data = self.__data[self.__stdate:self.__endate]

        # self.__data.reset_index(drop=True, inplace=True)

        # Rename columns with NHM HRU ids
        ren_dict = {v + 5: k for k, v in self.__indices.items()}

        # NOTE: The rename is an expensive operation
        self.__data.rename(columns=ren_dict, inplace=True)

    def read_cbh_full(self):
        """Read entire CBH file.
        """

        # incl_cols = list(self.__indices.values())
        # for xx in CBH_INDEX_COLS[:-1]:
        #     incl_cols.insert(0, xx)

        print('READING')
        # Columns 0-5 always represent date/time information
        self.__data = pd.read_csv(self.__src_path, sep=' ', skipinitialspace=True,
                                  skiprows=3, engine='c', memory_map=True,
                                  date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                  index_col='time', header=None, na_values=[-99.0, -999.0])

        if self.__stdate is not None and self.__endate is not None:
            self.__data = self.__data[self.__stdate:self.__endate]

        # self.__data.reset_index(drop=True, inplace=True)

        # Rename columns with NHM HRU ids
        # ren_dict = {v + 5: k for k, v in self.__indices.items()}

        # NOTE: The rename is an expensive operation
        # self.__data.rename(columns=ren_dict, inplace=True)
        self.__data['year'] = self.__data.index.year
        self.__data['month'] = self.__data.index.month
        self.__data['day'] = self.__data.index.day
        self.__data['hour'] = 0
        self.__data['minute'] = 0
        self.__data['second'] = 0

    def read_ascii_file(self, filename: str, columns=None) -> pd.DataFrame:
        """Reads a single CBH file.

        :param str filename: name of the CBH file
        :param columns: columns to read
        :type columns: None or """
        # Columns 0-5 always represent date/time information
        if columns is not None:
            df = pd.read_csv(filename, sep=' ', skipinitialspace=True,
                             usecols=columns,
                             skiprows=3, engine='c', memory_map=True,
                             date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                             index_col='time', header=None, na_values=[-99.0, -999.0, 'NaN', 'inf'])
        else:
            df = pd.read_csv(filename, sep=' ', skipinitialspace=True,
                             skiprows=3, engine='c', memory_map=True,
                             date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                             index_col='time', header=None, na_values=[-99.0, -999.0, 'NaN', 'inf'])
        return df

    def check_region(self, region: str) -> Union[OrderedDictType[int, int], None]:
        if self.__indices is not None:
            # Get the range of nhm_ids for the region
            rvals = self.__mapping[region]

            # print('Examining {} ({} to {})'.format(rr, rvals[0], rvals[1]))
            if rvals[0] >= rvals[1]:
                raise ValueError('Lower HRU bound is greater than upper HRU bound.')

            idx_retrieve = OrderedDict()

            for yy in self.__indices.keys():
                if rvals[0] <= yy <= rvals[1]:
                    idx_retrieve[self.__indices[yy]] = yy  # {local_ids: nhm_ids}

            return idx_retrieve
        return None

    def read_cbh_multifile(self, var=None) -> Union[pd.DataFrame, None]:
        """Read cbh data from multiple csv files"""

        if var is None:
            raise ValueError('Variable name (var) must be provided')

        first = True
        self.__dataframe = None

        for rr in REGIONS:
            idx_retrieve = self.check_region(region=rr)

            if len(idx_retrieve) > 0:
                # Build the list of columns to load
                # The given local ids must be adjusted by 5 to reflect:
                #     1) the presence of 6 columns of time information
                #     2) 0-based column names
                load_cols = list(CBH_INDEX_COLS)
                load_cols.extend([xx+5 for xx in idx_retrieve.keys()])
            else:
                load_cols = None

            if len(idx_retrieve) > 0:
                # The current region contains HRUs in the model subset
                # Read in the data for those HRUs
                cbh_file = f'{self.__src_path}/{rr}_{var}.cbh.gz'

                print(f'\tLoad {len(idx_retrieve)} HRUs from {rr}')

                if not os.path.isfile(cbh_file):
                    # Missing data file for this variable and region
                    raise IOError(f'Required CBH file, {cbh_file}, is missing.')

                # df = self.read_ascii_file(cbh_file, columns=load_cols)

                # Small read to get number of columns
                df = pd.read_csv(cbh_file, sep=' ', skipinitialspace=True,
                                 usecols=load_cols, nrows=2,
                                 skiprows=3, engine='c', memory_map=True,
                                 date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                 index_col='time', header=None, na_values=[-99.0, -999.0, 'NaN', 'inf'])

                # Override Pandas' rather stupid default of float64
                col_dtypes = {xx: np.float32 for xx in df.columns}

                # Now read the whole file using float32 instead of float64
                df = pd.read_csv(cbh_file, sep=' ', skipinitialspace=True,
                                 usecols=load_cols, dtype=col_dtypes,
                                 skiprows=3, engine='c', memory_map=True,
                                 date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                 index_col='time', header=None, na_values=[-99.0, -999.0, 'NaN', 'inf'])

                if self.__stdate is not None and self.__endate is not None:
                    # Restrict the date range
                    df = df[self.__stdate:self.__endate]

                # Rename columns with NHM HRU ids
                ren_dict = {k+5: v for k, v in idx_retrieve.items()}

                # NOTE: The rename is an expensive operation
                df.rename(columns=ren_dict, inplace=True)

                if first:
                    self.__dataframe = df.copy()
                    first = False
                else:
                    self.__dataframe = self.__dataframe.join(df, how='left')
        return self.__dataframe

    def get_var(self, var: str) -> Union[pd.DataFrame, None]:
        data = self.read_cbh_multifile(var=var)
        return data

    def write_ascii(self, pathname=None, fileprefix=None, variables=None):
        # For out_order the first six columns contain the time information and
        # are always output for the cbh files
        out_order = [kk for kk in self.__nhm_hrus]
        for cc in ['second', 'minute', 'hour', 'day', 'month', 'year']:
            out_order.insert(0, cc)

        var_list = []
        if variables is None:
            var_list = CBH_VARNAMES
        elif isinstance(list, variables):
            var_list = variables

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
                outfile = f'{cvar}.cbh'
            else:
                outfile = f'{fileprefix}_{cvar}.cbh'

            if pathname is not None:
                outfile = f'{pathname}/{outfile}'

            out_cbh = open(outfile, 'w')
            out_cbh.write('Written by Bandit\n')
            out_cbh.write(f'{cvar} {len(self.__nhm_hrus)}\n')
            out_cbh.write('########################################\n')

            data.to_csv(out_cbh, columns=out_order, na_rep='-999', float_format='%0.3f',
                        sep=' ', index=False, header=False, encoding=None, chunksize=50)
            out_cbh.close()

    def write_netcdf(self, filename=None, variables=None):
        """Write CBH to netcdf format file"""

        # NetCDF-related variables
        var_desc = {'tmax': 'Maximum Temperature', 'tmin': 'Minimum temperature', 'prcp': 'Precipitation'}
        var_units = {'tmax': 'C', 'tmin': 'C', 'prcp': 'inches'}

        # Create a netCDF file for the CBH data
        nco = nc.Dataset(filename, 'w', clobber=True)
        nco.createDimension('hru', len(self.__nhm_hrus))
        nco.createDimension('time', None)

        timeo = nco.createVariable('time', 'f4', ('time'))
        timeo.calendar = 'standard'
        # timeo.bounds = 'time_bnds'

        # FIXME: Days since needs to be set to the starting date of the model pull
        timeo.units = 'days since 1980-01-01 00:00:00'

        hruo = nco.createVariable('hru', 'i4', ('hru'))
        hruo.long_name = 'Hydrologic Response Unit ID (HRU)'

        var_list = []
        if variables is None:
            var_list = CBH_VARNAMES
        elif isinstance(list, variables):
            var_list = variables

        for cvar in var_list:
            varo = nco.createVariable(cvar, 'f4', ('time', 'hru'), fill_value=nc.default_fillvals['f4'], zlib=True)
            varo.long_name = var_desc[cvar]
            varo.units = var_units[cvar]

        nco.setncattr('Description', 'Climate by HRU')
        # nco.setncattr('Bandit_version', __version__)
        # nco.setncattr('NHM_version', nhmparamdb_revision)

        # Write the HRU ids
        hruo[:] = self.__nhm_hrus

        first = True
        for cvar in var_list:
            data = self.get_var(var=cvar)

            if first:
                timeo[:] = nc.date2num(data.index.tolist(),
                                       units='days since 1980-01-01 00:00:00',
                                       calendar='standard')
                first = False

            # Write the CBH values
            nco.variables[cvar][:, :] = data[self.__nhm_hrus].values

        nco.close()

    # def write_cbh_subset(self, outdir):
    #     outdata = None
    #     first = True
    #
    #     for vv in CBH_VARNAMES:
    #         outorder = list(CBH_INDEX_COLS)
    #
    #         for rr, rvals in iteritems(self.__mapping):
    #             idx_retrieve = {}
    #
    #             for yy in self.__nhm_hrus.keys():
    #                 if rvals[0] <= yy <= rvals[1]:
    #                     idx_retrieve[yy] = self.__nhm_hrus[yy]
    #
    #             if len(idx_retrieve) > 0:
    #                 self.__src_path = '{}/{}_{}.cbh.gz'.format(self.__cbhdb_dir, rr, vv)
    #                 self.read_cbh()
    #                 if first:
    #                     outdata = self.__data
    #                     first = False
    #                 else:
    #                     outdata = pd.merge(outdata, self.__data, how='left', left_index=True, right_index=True)
    #
    #         # Append the HRUs as ordered for the subset
    #         outorder.extend(self.__nhm_hrus)
    #
    #         out_cbh = open('{}/{}.cbh'.format(outdir, vv), 'w')
    #         out_cbh.write('Written by pyPRMS.Cbh\n')
    #         out_cbh.write('{} {}\n'.format(vv, len()))

    # def read_cbh_parq(self, src_dir):
    #     """Read CBH files stored in the parquet format"""
    #     if self.__indices:
    #         pfile = fp.ParquetFile('{}/daymet_{}.parq'.format(src_dir, self.__var))
    #         self.__data = pfile.to_pandas(self.__indices)
    #
    #         if self.__stdate is not None and self.__endate is not None:
    #             # Given a date range to restrict the output
    #             self.__data = self.__data[self.__stdate:self.__endate]
    #
    #     self.__data['year'] = self.__data.index.year
    #     self.__data['month'] = self.__data.index.month
    #     self.__data['day'] = self.__data.index.day
    #     self.__data['hour'] = 0
    #     self.__data['minute'] = 0
    #     self.__data['second'] = 0
    #
    # def read_cbh_hdf(self, src_dir):
    #     """Read CBH files stored in HDF5 format"""
    #     if self.__indices:
    #         # self.__data = pd.read_hdf('{}/daymet_{}.h5'.format(src_dir, self.__var), columns=self.__indices)
    #         self.__data = pd.read_hdf('{}/daymet_{}.h5'.format(src_dir, self.__var))
    #
    #         if self.__stdate is not None and self.__endate is not None:
    #             # Given a date range to restrict the output
    #             self.__data = self.__data[self.__stdate:self.__endate]
    #
    #         self.__data = self.__data[self.__indices]
    #
    #     self.__data['year'] = self.__data.index.year
    #     self.__data['month'] = self.__data.index.month
    #     self.__data['day'] = self.__data.index.day
    #     self.__data['hour'] = 0
    #     self.__data['minute'] = 0
    #     self.__data['second'] = 0
    #
