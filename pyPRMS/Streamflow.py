import re
import numpy as np
import pandas as pd   # type: ignore

# TS_FORMAT = '%Y %m %d %H %M %S'   # 1915 1 13 0 0 0


class Streamflow(object):
    """Class for working with observed streamflow in the PRMS ASCII data file format"""

    def __init__(self, filename, missing=-999.0, verbose=False, include_metadata=True):

        self.__missing = missing
        self.filename = filename
        self.__verbose = verbose
        self.__include_metadata = include_metadata

        self.__timecols = 6  # number columns for time in the file
        self.__headercount = None
        self.__metaheader = None
        self.__types = None
        self.__units = {}
        self.__stations = []
        self.__stationIndex = {}  # Lookup of station id to header info
        self.__rawdata = None
        self.__selectedStations = None
        self.__isloaded = False

        self.load_file(self.filename)

    @property
    def data(self):
        """Pandas dataframe of the observed streamflow for each POI"""

        if self.__selectedStations is None:
            return self.__rawdata
        else:
            return self.__rawdata.ix[:, self.__selectedStations]

    @property
    def headercount(self):
        """Number of rows to skip before data begins"""

        return self.__headercount

    @property
    def metaheader(self):
        """List of columns in the metadata section of the data file"""

        return self.__metaheader

    @property
    def numdays(self):
        """The number of days in the period of record"""

        return self.data.shape[0]

    @property
    def size(self):
        """Number of streamgages"""
        return len(self.stations)

    @property
    def stations(self):
        """List of streamgage IDs from streamflow data file"""

        return self.__stations

    @property
    def units(self):
        """Dictionary of units for observation values"""
        return self.__units

    def load_file(self, filename):
        """Read the PRMS ASCII streamflow data file"""

        infile = open(filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        self.__headercount = 0

        # We assume if 'ID' and 'Type' header names exist then we have a valid
        # meta-header.
        for line in it:
            self.__headercount += 1

            # Skip lines until we hit the following
            if line[0:10] == '// Station':
                # Read the next line - these are the fieldnames for the station information
                self.__headercount += 1
                self.__metaheader = re.findall(r"[\w]+", next(it))
                break

        cnt = 0
        order = 0  # defines the order of the data types in the dataset
        st = 0

        # Read the station IDs and optional additional metadata
        for line in it:
            self.__headercount += 1

            if line[0:10] == '//////////':
                break

            # Read station information
            # Include question mark in regex below as a valid character since the obs
            # file uses it for missing data in the station information.
            words = re.findall(r"[\w.-]+|[?]", line)  # Break the row up
            curr_fcnt = len(words)

            # Check that number of station information fields remains constant
            if curr_fcnt != len(self.__metaheader):
                if self.__verbose:
                    print("WARNING: number of header fields changed from %d to %d" %
                          (len(self.__metaheader), curr_fcnt)),
                    print("\t", words)
                    # exit()

            try:
                if words[self.__metaheader.index('Type')] not in self.__types:
                    # Add unique station types (e.g. precip, runoff) if a 'Type' field exists in the metaheader
                    st = cnt  # last cnt becomes the starting column of the next type
                    order += 1

                # Information stored in __types array:
                # 1) Order that type was added in
                # 2) Starting index for data section
                # 3) Ending index for data section
                self.__types[words[self.__metaheader.index('Type')]] = [order, st, cnt]
            except ValueError:
                if self.__verbose:
                    print('No "Type" metadata; skipping.')

            self.__stations.append(words[0])
            self.__stationIndex[words[0]] = cnt
            cnt += 1

        # Read the units and add to each type
        unittmp = next(it).split(':')[1].split(',')
        self.__headercount += 1
        for xx in unittmp:
            unit_pair = xx.split('=')
            self.__units[unit_pair[0].strip()] = unit_pair[1].strip()

        # Skip to the data section
        for line in it:
            self.__headercount += 1
            if line[0:10] == '##########':
                break

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Data section
        # The first 6 columns are [year month day hour minute seconds]
        thecols = ['year', 'month', 'day', 'hour', 'minute', 'second']
        timecols = thecols.copy()

        # Add the remaining columns to the list
        for xx in self.__stations:
            thecols.append(xx)

        # Use pandas to read the data in from the remainder of the file
        # We use a custom date parser to convert the date information to a datetime
        # NOTE: 2023-03-21 skiprows option seems to be off by 1; test data starts
        #       at line 26, but skiprows=25 skips the first row of data.
        self.__rawdata = pd.read_csv(self.filename, skiprows=self.__headercount-1, sep=r'\s+',
                                     header=0, names=thecols, engine='c', skipinitialspace=True)

        self.__rawdata['time'] = pd.to_datetime(self.__rawdata[timecols], yearfirst=True)
        self.__rawdata.drop(columns=timecols, inplace=True)
        self.__rawdata.set_index('time', inplace=True)

        # Convert the missing data (-999.0) to NaNs
        self.__rawdata.replace(to_replace=self.__missing, value=np.nan, inplace=True)

        self.__isloaded = True

    # @property
    # def timecolcnt(self):
    #     return self.__timecols

    # @property
    # def types(self):
    #     if not self.__isloaded:
    #         self.load_file(self.filename)
    #     return self.__types

    # def get_data_by_type(self, thetype):
    #     """Returns data selected type (e.g. runoff)"""
    #
    #     if thetype in self.__types:
    #         # print "Selected type '%s':" % (thetype), self.__types[thetype]
    #         st = self.__types[thetype][1]
    #         en = self.__types[thetype][2]
    #         # print "From %d to %d" % (st, en)
    #         b = self.data.iloc[:, st:en + 1]
    #
    #         return b
    #     else:
    #         print("not found")
    #
    # def get_stations_by_type(self, thetype):
    #     """Returns station IDs for a given type (e.g. runoff)"""
    #
    #     if thetype in self.__types:
    #         # print "Selected type '%s':" % (thetype), self.__types[thetype]
    #         st = self.__types[thetype][1]
    #         en = self.__types[thetype][2]
    #         # print "From %d to %d" % (st, en)
    #         b = self.stations[st:en + 1]
    #
    #         return b
    #     else:
    #         print("not found")
    #
    # def select_by_station(self, streamgages):
    #     """Selects one or more streamgages from the dataset"""
    #     # The routine writeSelected() will write selected streamgages and data
    #     # to a new PRMS streamflow file.
    #     # Use clearSelectedStations() to clear any current selection.
    #     if isinstance(streamgages, list):
    #         self.__selectedStations = streamgages
    #     else:
    #         self.__selectedStations = [streamgages]
    #
    # def clear_selected_stations(self):
    #     """Clears any selected streamgages"""
    #     self.__selectedStations = None
    #
    # def write_selected_stations(self, filename):
    #     """Writes station observations to a new file"""
    #     # Either writes out all station observations or, if stations are selected,
    #     # then a subset of station observations.
    #
    #     # Sample header format
    #
    #     # $Id:$
    #     # ////////////////////////////////////////////////////////////
    #     # // Station metadata (listed in the same order as the data):
    #     # // ID    Type Latitude Longitude Elevation
    #     # // <station info>
    #     # ////////////////////////////////////////////////////////////
    #     # // Unit: runoff = ft3 per sec, elevation = feet
    #     # ////////////////////////////////////////////////////////////
    #     # runoff <number of stations for each type>
    #     # ################################################################################
    #
    #     top_line = '$Id:$\n'
    #     section_sep = '////////////////////////////////////////////////////////////\n'
    #     meta_header_1 = '// Station metadata (listed in the same order as the data):\n'
    #     # metaHeader2 = '// ID    Type Latitude Longitude Elevation'
    #     meta_header_2 = '// %s\n' % ' '.join(self.metaheader)
    #     data_section = '################################################################################\n'
    #
    #     # ----------------------------------
    #     # Get the station information for each selected station
    #     type_count = {}  # Counts the number of stations for each type of data (e.g. 'runoff')
    #     stninfo = ''
    #     if self.__selectedStations is None:
    #         for xx in self.__stations:
    #             if xx[1] not in type_count:
    #                 # index 1 should be the type field
    #                 type_count[xx[1]] = 0
    #             type_count[xx[1]] += 1
    #
    #             stninfo += '// %s\n' % ' '.join(xx)
    #     else:
    #         for xx in self.__selectedStations:
    #             cstn = self.__stations[self.__stationIndex[xx]]
    #
    #             if cstn[1] not in type_count:
    #                 # index 1 should be the type field
    #                 type_count[cstn[1]] = 0
    #
    #             type_count[cstn[1]] += 1
    #
    #             stninfo += '// %s\n' % ' '.join(cstn)
    #     # stninfo = stninfo.rstrip('\n')
    #
    #     # ----------------------------------
    #     # Get the units information
    #     unit_line = '// Unit:'
    #     for uu in self.__units:
    #         unit_line += ' %s,' % ' = '.join(uu)
    #     unit_line = '%s\n' % unit_line.rstrip(',')
    #
    #     # ----------------------------------
    #     # Create the list of types of data that are being included
    #     tmpl = []
    #
    #     # Create list of types in the correct order
    #     for (kk, vv) in self.__types.items():
    #         if kk in type_count:
    #             tmpl.insert(vv[0], [kk, type_count[kk]])
    #
    #     type_line = ''
    #     for tt in tmpl:
    #         type_line += '%s %d\n' % (tt[0], tt[1])
    #     # typeLine = typeLine.rstrip('\n')
    #
    #     # Write out the header to the new file
    #     outfile = open(filename, 'w')
    #     outfile.write(top_line)
    #     outfile.write(section_sep)
    #     outfile.write(meta_header_1)
    #     outfile.write(meta_header_2)
    #     outfile.write(stninfo)
    #     outfile.write(section_sep)
    #     outfile.write(unit_line)
    #     outfile.write(section_sep)
    #     outfile.write(type_line)
    #     outfile.write(data_section)
    #
    #     # Write out the data to the new file
    #     # Using quoting=csv.QUOTE_NONE results in an error when using a customized  date_format
    #     # A kludgy work around is to write with quoting and then re-open the file
    #     # and write it back out, stripping the quote characters.
    #     self.data.to_csv(outfile, index=True, header=False, date_format='%Y %m %d %H %M %S', sep=' ')
    #     outfile.close()
    #
    #     old = open(filename, 'r').read()
    #     new = re.sub('["]', '', old)
    #     open(filename, 'w').write(new)
    #
    #     # def getRecurrenceInterval(self, thetype):
    #     #     """Returns the recurrence intervals for each station"""
    #     #
    #     #     # Copy the subset of data
    #     #     xx = self.seldata(thetype)
    #     #
    #     #     ri = np.zeros(xx.shape)
    #     #     ri[:,:] = -1.
    #     #
    #     #     # for each station we need to compute the RI for non-zero values
    #     #     for ss in range(0,xx.shape[1]):
    #     #         tmp = xx[:,ss]              # copy values for current station
    #     #
    #     #         # Get array of indices that would result in a sorted array
    #     #         sorted_ind = np.argsort(tmp)
    #     #         #print "sorted_ind.shape:", sorted_ind.shape
    #     #
    #     #         numobs = tmp[(tmp > 0.0),].shape[0]  # Number of observations > 0.
    #     #         nyr = float(numobs / 365)     # Number of years of non-zero observations
    #     #
    #     #         nz_cnt = 0  # non-zero value counter
    #     #         for si in sorted_ind:
    #     #             if tmp[si] > 0.:
    #     #                 nz_cnt += 1
    #     #                 rank = numobs - nz_cnt + 1
    #     #                 ri[si,ss] = (nyr + 1.) / float(rank)
    #     #                 #print "%s: [%d]: %d %d %0.3f %0.3f" % (ss, si,  numobs, rank, tmp[si], ri[si,ss])
    #     #
    #     #     return ri
# ***** END of class streamflow()
