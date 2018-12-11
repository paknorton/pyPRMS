from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import os
import pandas as pd
# import fastparquet as fp
# import xarray as xr
from collections import OrderedDict

from pyPRMS.prms_helpers import dparse
from pyPRMS.constants import REGIONS

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class Cbh(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-12-05
    # Description: Class for working with individual cbh files
    #
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

    def __init__(self, filename=None, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None,
                 var=None, regions=REGIONS):
        # def __init__(self, cbhdb_dir, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None):
        #     self.__cbhdb_dir = cbhdb_dir
        self.__filename = filename

        # self.__indices = [str(kk) for kk in indices]
        self.__indices = indices    # OrdereDict: nhm_ids -> local_ids

        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus
        self.__mapping = mapping
        self.__date_range = None
        self.__data = None
        self.__final_outorder = None
        self.__var = var
        self.__regions = regions

    @property
    def data(self):
        return self.__data

    def read_cbh(self):
        # Reads a full cbh file

        # incl_cols = list(self.__indices.values())
        # for xx in CBH_INDEX_COLS[:-1]:
        #     incl_cols.insert(0, xx)

        incl_cols = list(CBH_INDEX_COLS)

        for xx in self.__indices.values():
            incl_cols.append(xx+5)  # include an offset for the having datetime info
        # print(incl_cols)

        # Columns 0-5 always represent date/time information
        self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True, usecols=incl_cols,
                                  skiprows=3, engine='c', memory_map=True,
                                  date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                  index_col='time', header=None, na_values=[-99.0, -999.0])

        if self.__stdate is not None and self.__endate is not None:
            self.__data = self.__data[self.__stdate:self.__endate]

        # self.__data.reset_index(drop=True, inplace=True)

        # Rename columns with NHM HRU ids
        ren_dict = {v + 5: k for k, v in iteritems(self.__indices)}
        # ren_dict = {v + 5: '{}'.format(k) for k, v in self.__indices.iteritems()}

        # NOTE: The rename is an expensive operation
        self.__data.rename(columns=ren_dict, inplace=True)

    def read_cbh_full(self):
        # incl_cols = list(self.__indices.values())
        # for xx in CBH_INDEX_COLS[:-1]:
        #     incl_cols.insert(0, xx)

        print('READING')
        # Columns 0-5 always represent date/time information
        self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True,
                                  skiprows=3, engine='c', memory_map=True,
                                  date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                  index_col='time', header=None, na_values=[-99.0, -999.0])

        if self.__stdate is not None and self.__endate is not None:
            self.__data = self.__data[self.__stdate:self.__endate]

        # self.__data.reset_index(drop=True, inplace=True)

        # Rename columns with NHM HRU ids
        # ren_dict = {v + 5: k for k, v in iteritems(self.__indices)}
        # ren_dict = {v + 5: '{}'.format(k) for k, v in self.__indices.iteritems()}

        # NOTE: The rename is an expensive operation
        # self.__data.rename(columns=ren_dict, inplace=True)
        self.__data['year'] = self.__data.index.year
        self.__data['month'] = self.__data.index.month
        self.__data['day'] = self.__data.index.day
        self.__data['hour'] = 0
        self.__data['minute'] = 0
        self.__data['second'] = 0

    def read_cbh_multifile(self, src_dir):
        """Read cbh data from multiple csv files"""
        first = True

        for rr in self.__regions:
            rvals = self.__mapping[rr]

            # print('Examining {} ({} to {})'.format(rr, rvals[0], rvals[1]))
            if rvals[0] >= rvals[1]:
                raise ValueError('Lower HRU bound is greater than upper HRU bound.')

            idx_retrieve = OrderedDict()

            for yy in self.__indices.keys():
                if rvals[0] <= yy <= rvals[1]:
                    # print('\tMatching region {}, HRU: {} ({})'.format(rr, yy, hru_order_ss[yy]))
                    idx_retrieve[self.__indices[yy]] = yy   # {local_ids: nhm_ids}

            if len(idx_retrieve) > 0:
                # The current region contains HRUs in the model subset
                # Read in the data for those HRUs
                cbh_file = '{}/{}_{}.cbh.gz'.format(src_dir, rr, self.__var)

                print('\tLoad {} HRUs from {}'.format(len(idx_retrieve), rr))

                if not os.path.isfile(cbh_file):
                    # Missing data file for this variable and region
                    # bandit_log.error('Required CBH file, {}, is missing. Unable to continue'.format(cbh_file))
                    raise IOError('Required CBH file, {}, is missing.'.format(cbh_file))

                # Build the list of columns to load
                # The given local ids must be adjusted by 5 to reflect:
                #     1) the presence of 6 columns of time information
                #     2) 0-based column names
                load_cols = list(CBH_INDEX_COLS)
                load_cols.extend([xx+5 for xx in idx_retrieve.keys()])

                # Columns 0-5 always represent date/time information
                df = pd.read_csv(cbh_file, sep=' ', skipinitialspace=True,
                                 usecols=load_cols,
                                 skiprows=3, engine='c', memory_map=True,
                                 date_parser=dparse, parse_dates={'time': CBH_INDEX_COLS},
                                 index_col='time', header=None, na_values=[-99.0, -999.0])

                if self.__stdate is not None and self.__endate is not None:
                    # Restrict the date range
                    df = df[self.__stdate:self.__endate]

                # Rename columns with NHM HRU ids
                ren_dict = {k+5: v for k, v in iteritems(idx_retrieve)}

                # NOTE: The rename is an expensive operation
                df.rename(columns=ren_dict, inplace=True)

                if first:
                    self.__data = df.copy()
                    first = False
                else:
                    self.__data = self.__data.join(df, how='left')
                    # outdata = pd.merge(outdata, cc1.data, on=[0, 1, 2, 3, 4, 5])

        self.__data['year'] = self.__data.index.year
        self.__data['month'] = self.__data.index.month
        self.__data['day'] = self.__data.index.day
        self.__data['hour'] = 0
        self.__data['minute'] = 0
        self.__data['second'] = 0

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
    #                 self.__filename = '{}/{}_{}.cbh.gz'.format(self.__cbhdb_dir, rr, vv)
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
    # def read_cbh_netcdf(self, src_dir):
    #     """Read CBH files stored in netCDF format"""
    #     if self.__indices:
    #         print('\t\tOpen dataarray')
    #         ds = xr.open_dataarray('{}/daymet_{}.nc'.format(src_dir, self.__var), chunks={'hru': 1000})
    #
    #         print('\t\tConvert subset to pandas dataframe')
    #         self.__data = ds.loc[:, self.__indices].to_pandas()
    #
    #         print('\t\tRestrict to date range')
    #         if self.__stdate is not None and self.__endate is not None:
    #             # Restrict dataframe to the given date range
    #             self.__data = self.__data[self.__stdate:self.__endate]
    #
    #         # self.__data = self.__data[self.__indices]
    #
    #     print('\t\tInsert date info')
    #     # self.__data.insert(0, 'second', 0)
    #     # self.__data.insert(0, 'minute', 0)
    #     # self.__data.insert(0, 'hour', 0)
    #     # self.__data.insert(0, 'day', self.__data.index.day)
    #     # self.__data.insert(0, 'month', self.__data.index.month)
    #     # self.__data.insert(0, 'year', self.__data.index.year)
    #     self.__data['year'] = self.__data.index.year
    #     self.__data['month'] = self.__data.index.month
    #     self.__data['day'] = self.__data.index.day
    #     self.__data['hour'] = 0
    #     self.__data['minute'] = 0
    #     self.__data['second'] = 0
