from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import pandas as pd
from pyPRMS.prms_helpers import dparse

CBH_VARNAMES = ['prcp', 'tmin', 'tmax']
CBH_INDEX_COLS = [0, 1, 2, 3, 4, 5]


class Cbh(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-12-05
    # Description: Class for working with individual cbh files
    #
    # This class assumes it is dealing with regional cbh files (not a CONUS-level NHM file)
    # TODO: As written type of data (e.g. tmax, tmin, prcp) is ignored.
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

    def __init__(self, filename=None, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None):
        # def __init__(self, cbhdb_dir, st_date=None, en_date=None, indices=None, nhm_hrus=None, mapping=None):
        #     self.__cbhdb_dir = cbhdb_dir
        self.__filename = filename
        self.__indices = indices    # This should be an ordered dict: nhm->local hrus
        self.__stdate = st_date
        self.__endate = en_date
        self.__nhm_hrus = nhm_hrus
        self.__mapping = mapping
        self.__date_range = None
        self.__data = None
        self.__final_outorder = None

    @property
    def data(self):
        return self.__data

    # def get_cbh_subset(self):
    #     # Get a subset of CBH values given a date range and HRUs
    #     # The subset can span over multiple regions

    def read_cbh(self):
        # Read the data
        if self.__indices:
            incl_cols = [0, 1, 2, 3, 4, 5]
            for xx in self.__indices.values():
                incl_cols.append(xx+5)  # include an offset for the having datetime info

            # Columns 0-5 always represent date/time information
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True, usecols=incl_cols,
                                      skiprows=3, engine='c', header=None)

            # Rename columns with NHM HRU ids
            ren_dict = {v+5: k for k, v in self.__indices.iteritems()}
            # print(ren_dict)

            # NOTE: The rename is an expensive operation
            self.__data.rename(columns=ren_dict, inplace=True)
        else:
            # Read the entire file
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True,
                                      skiprows=3, engine='c', header=None)
        # in_hdl.close()

    def read_cbh2(self):
        if self.__indices:
            incl_cols = CBH_INDEX_COLS.copy()
            for xx in self.__indices.values():
                incl_cols.append(xx+5)  # include an offset for the having datetime info

            # Columns 0-5 always represent date/time information
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True, usecols=incl_cols,
                                      skiprows=3, engine='c',
                                      date_parser=dparse, parse_dates={'thedate': [0, 1, 2, 3, 4, 5]},
                                      index_col='thedate', header=None, na_values=[-99.0, -999.0])
            print(self.__data.head())

            # Rename columns with NHM HRU ids
            ren_dict = {v + 5: k for k, v in self.__indices.iteritems()}
            # print(ren_dict)

            # NOTE: The rename is an expensive operation
            self.__data.rename(columns=ren_dict, inplace=True)
        else:
            # NOTE: This is incomplete and won't work properly without code to rename to columns
            #       to national HRU ids.

            # Load the CBH file
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True, skiprows=3, engine='c',
                                      date_parser=dparse, parse_dates={'thedate': [0, 1, 2, 3, 4, 5]},
                                      index_col='thedate', header=None, na_values=[-99.0, -999.0])

    def write_cbh_subset(self, outdir):
        outdata = None
        first = True

        for vv in CBH_VARNAMES:
            outorder = CBH_INDEX_COLS.copy()

            for rr, rvals in iteritems(self.__mapping):
                idx_retrieve = {}

                for yy in self.__nhm_hrus.keys():
                    if rvals[0] <= yy <= rvals[1]:
                        idx_retrieve[yy] = self.__nhm_hrus[yy]

                if len(idx_retrieve) > 0:
                    self.__filename = '{}/{}_{}.cbh.gz'.format(self.__cbhdb_dir, rr, vv)
                    self.read_cbh2()
                    if first:
                        outdata = self.__data
                        first = False
                    else:
                        outdata = pd.merge(outdata, self.__data, how='left', left_index=True, right_index=True)

            # Append the HRUs as ordered for the subset
            outorder.extend(self.__nhm_hrus)

            out_cbh = open('{}/{}.cbh'.format(outdir, vv), 'w')
            out_cbh.write('Written by Skein\n')
            out_cbh.write('{} {}\n'.format(vv, len()))
