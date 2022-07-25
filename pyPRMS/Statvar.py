
import numpy as np
import pandas as pd   # type: ignore

# from pyPRMS.prms_helpers import dparse

TS_FORMAT = '%Y %m %d %H %M %S' # 1915 1 13 0 0 0

class Statvar(object):
    def __init__(self, filename=None, missing=-999.0):
        self.__timecols = 6  # number columns for time in the file
        self.__missing = missing  # what is considered a missing value?

        self.__isloaded = False
        self.__vars = None
        self.__rawdata = None
        self.__metaheader = None
        self.__header = None
        self.__headercount = None
        self.__types = None
        self.__units = {}
        self.__stations = None
        self.__filename = ''
        self.filename = filename  # trigger the filename setter

    @property
    def filename(self):
        if not self.__isloaded:
            self.load_file(self.__filename)
        return self.__filename

    @filename.setter
    def filename(self, fname):
        self.__isloaded = False
        self.__vars = None
        self.__rawdata = None
        self.__metaheader = None
        self.__header = None
        self.__headercount = None
        self.__types = None
        self.__units = {}
        self.__stations = None
        self.__filename = fname

        self.load_file(self.__filename)

    def load_file(self, filename):
        """Load a statvar file"""

        infile = open(filename, 'r')

        # The first line gives the number of variables that follow
        numvars = int(infile.readline())
        # print "Number of variables: %d" % (numvars)

        # The next numvar rows contain a variable name followed by a number which
        # indicates the number of columns used by that variable.
        # The relative order of the variable in the list indicates the column
        # the variable data is found in.
        self.__isloaded = False
        self.__vars = {}
        self.__header = []

        # The first 7 columns are [record year month day hour minute seconds]
        self.__header = ['rec', 'year', 'month', 'day', 'hour', 'min', 'sec']

        for rr in range(0, numvars):
            row = infile.readline()
            fields = row.rstrip().split(' ')
            varname = fields[0]
            varsize = int(fields[1])

            # Store the variable name along with the order it was read
            # and the dimension size of the variable.
            self.__vars[varname] = [rr, varsize]

            # Add to the header
            # TODO: Lookup each variable to find the dimension name.
            #       This could be used to create informative headers
            for dd in range(0, varsize):
                if varsize > 1:
                    # If a variable has dimension more than one (e.g. hru)
                    # then append a sequential number to each instance
                    self.__header.append('%s_%d' % (varname, dd + 1))
                else:
                    self.__header.append('%s' % varname)

        # Now load the data

        # Use pandas to read the data in from the remainder of the file
        # We use a custom date parser to convert the date information to a datetime
        self.__rawdata = pd.read_csv(infile, sep=r"\s+", header=None, names=self.__header,
                                     parse_dates={'time': ['year', 'month', 'day', 'hour', 'min', 'sec']},
                                     index_col='time')
                                     # date_parser=dparse, index_col='time')

        self.__rawdata.index = pd.to_datetime(self.__rawdata.index, exact=True, cache=True, format=TS_FORMAT)

        # Drop the 'rec' field and convert the missing data to NaNs
        self.__rawdata.drop(['rec'], axis=1, inplace=True)
        self.__rawdata.replace(to_replace=self.__missing, value=np.nan, inplace=True)

        self.__isloaded = True

    # **** END def load_file()

    @property
    def headercount(self):
        """Returns the size of the header list"""
        if not self.__isloaded:
            self.load_file(self.filename)
        return len(self.__header)

    @property
    def vars(self):
        """Returns a dictionary of the variables in the statvar file"""
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__vars

    @property
    def data(self):
        """Returns the pandas dataframe of data from the statvar file"""
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__rawdata

# ***** END of class statvar()
