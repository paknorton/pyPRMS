#!/usr/bin/env python
import re
import numpy as np
import pandas as pd
import datetime
import sys


# Author: Parker Norton (pnorton@usgs.gov)
# Create date: 2015-02-09
# Description: Set of classes for processing PRMS data files. The datafiles
#              that are currently handled are:
#                  streamflow - processes the streamflow data
#                  param - processes the input parameter file
#                  statvar - processes the output statvar file
#                  control - processes the control file

__version__ = '0.5'


def dparse(yr, mo, dy, hr, minute, sec):
    # Date parser for working with the date format from PRMS files

    # Convert to integer first
    yr, mo, dy, hr, minute, sec = [int(x) for x in [yr, mo, dy, hr, minute, sec]]

    dt = datetime.datetime(yr, mo, dy, hr, minute, sec)
    return dt


def create_default_range_file(in_filename, out_filename):
    """Get the default parameter ranges from a file which is the result of running
       'prms -print'"""

    # Create parameter default ranges file from from PRMS -print results
    try:
        infile = open(in_filename, 'r')
    except IOError as err:
        print "Unable to open file\n", err
        return False
    else:
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        for line in it:
            if line == '--------------- PARAMETERS ---------------':
                break
        param_dict = {}

        for line in it:
            flds = line.split(':')

            if len(flds) < 2:
                continue

            key = flds[0].strip()
            val = flds[1].strip()

            if key == 'Name':
                cparam = val
                param_dict[cparam] = {}
            else:
                param_dict[cparam][key] = val

        try:
            outfile = open(out_filename, 'w')
        except IOError as err:
            print "Unable to write file\n", err
            return False
        else:
            outfile.write('parameter max min\n')

            for kk, vv in param_dict.iteritems():
                outfile.write('%s %f %f\n' % (kk, float(vv['Max']), float(vv['Min'])))

            outfile.close()
        return True


def to_datetime(date_str):
    """Takes a date string of the form 'YYYY-MM-DD HH:mm:ss' (and variations thereof)
       and converts it to a datetime"""
    return datetime.datetime(*[int(x) for x in re.split('-| |:', date_str)])


def to_prms_datetime(date):
    """Takes a datetime object and converts it to a string of form
       YYYY,MM,DD,HH,mm,ss"""
    return date.strftime('%Y,%m,%d,%H,%M,%S')

# Order to write control file parameters for printing and writing a new control file
ctl_order = ['start_time', 'end_time', 'print_debug', 'executable_desc', 'executable_model', 'model_mode',
             'et_module', 'precip_module', 'soilzone_module', 'solrad_module', 'srunoff_module',
             'strmflow_module', 'temp_module', 'transp_module',
             'cascade_flag', 'cascadegw_flag', 'cbh_check_flag',
             'dprst_flag', 'dyn_snareathresh_flag', 'parameter_check_flag',
             'param_file', 'data_file', 'humidity_day', 'orad_flag', 'potet_day',
             'precip_day', 'swrad_day', 'tmax_day', 'tmin_day', 'transp_day', 'windspeed_day', 'model_output_file',
             'csvON_OFF', 'csv_output_file',
             'nhruOutBase_FileName', 'nhruOutON_OFF', 'nhruOutVar_names', 'nhruOutVars',
             'nstatVars', 'statVar_element', 'statVar_names', 'stat_var_file', 'statsON_OFF', 'stats_output_file',
             'aniOutON_OFF', 'aniOutVar_names', 'ani_output_file', 'naniOutVars',
             'dispGraphsBuffSize', 'dispVar_element', 'dispVar_names', 'dispVar_plot', 'initial_deltat', 'ndispGraphs',
             'nmapOutVars', 'mapOutON_OFF', 'mapOutVar_names',
             'init_vars_from_file', 'save_vars_to_file', 'var_init_file', 'var_save_file']

class control(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2015-02-05
    # Description: Class object to handle reading and writing the PRMS
    #              control files.

    def __init__(self, filename):
        # 1) open file
        # 2) read file contents

        self.__isloaded = False
        self.__filename = filename
        self.__rowdelim = '####'    # Used to delimit variables
        self.__valtypes = ['', 'integer', 'float', 'double', 'string']

        self.filename = filename
    # END __init__

    def __getattr__(self, item):
        return self.get_var(item)

    def __str__(self):
        outstr = ''
        for xx in ctl_order:
            try:
                pp = self.__controldict[xx]
                if len(pp['values']) == 1:
                    outstr += '%s: %s, %s\n' % (xx, self.__valtypes[pp['valuetype']], str(pp['values'][0]))
                else:
                    outstr += '%s: %s, %d values\n' % (xx, self.__valtypes[pp['valuetype']], len(pp['values']))
            except:
                continue
        return outstr

    @property
    def filename(self):
        if not self.__isloaded:
            self.load_file(self.__filename)
        return self.__filename

    @filename.setter
    def filename(self, fname):
        self.__isloaded = False

        self.__controldict = {}
        self.__header = []

        self.__filename = fname

        self.load_file(self.__filename)

    @property
    def vars(self):
        if not self.__isloaded:
            self.load_file(self.__filename)

        varlist = []

        for cc in self.__controldict:
            varlist.append(cc)
        return varlist

    @property
    def rawvars(self):
        return self.__controldict

    def load_file(self, filename):
        # Read the control file into memory and parse it
        self.__isloaded = False
        self.__controldict = {}   # Initialize the control dictionary

        infile = open(filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        for line in it:
            if line[0:4] == '$Id:':
                self.__header.append(line)
                continue
            elif line == self.__rowdelim:
                continue
            else:
                # We're dealing with a control parameter/variable
                # We're in a parameter section
                vardict = {}    # temporary to build variable info
                varname = line.split(' ')[0]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check for duplicate variable name
                if varname in self.__controldict:
                    # Check for duplicate variables (that couldn't happen! :))
                    # If it does skip to the next variable in the parameter file
                    print 'Duplicate variable name, %s, in Parameters section.. skipping' \
                              % varname

                    try:
                        while next(it) != self.__rowdelim:
                            pass
                    except StopIteration:
                        # We hit the end of the file
                        continue
                    continue
                    # END check for duplicate varnames
                else:
                    # Add variable to dictionary
                    self.__controldict[varname] = {}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                numval = int(next(it))    # number of values for this variable
                valuetype = int(next(it))    # Variable type (1 - integer, 2 - float, 4 - character)
                # print '\tnumval:', numval
                # print '\tvaluetype:', valuetype

                vardict['valuetype'] = int(valuetype)
                vals = []

                for vv in range(0, numval):
                    try:
                        if valuetype == 1:  # integer
                            vals.append(int(next(it)))
                        elif valuetype == 2:    # float
                            vals.append(float(next(it)))
                        else:   # character
                            vals.append(next(it))
                    except ValueError:
                        print "varname: %s value type and defined type (%s) don't match" \
                                % (varname, self.__valtypes[valuetype])
                vardict['values'] = vals

                if len(vals) != numval:
                    print 'ERROR: Not enough values provided for %s' % (varname)
                    print '       Expect %d, got %d' % (numval, len(vals))

                # Check if there are too many values specified
                try:
                    cnt = numval
                    while next(it) != self.__rowdelim:
                        cnt +=1

                    if cnt > numval:
                        print 'WARNING: Too many values specified for %s' % varname
                        print '       %d expected, %d given' % (numval, cnt)
                        print '       Keeping first %d values' % (numval)
                except StopIteration:
                    # Hit the end of the file
                    pass
                    # continue
                # self.__controldict[varname].append(vardict)
                self.__controldict[varname] = vardict
        # ***** END for line in it
        self.__isloaded = True
    # END **** load_file()

    def clear_parameter_group(self, grpname):
        """Given a single parameter group name will clear out values for that parameter
           and all related parameters. Group name is one of: statVar, ani, map, dispVar, nhru"""

        grp_name = {'ani': {'naniOutVars': 0, 'aniOutON_OFF': 0, 'aniOutVar_names': []},
                    'dispVar': {'ndispGraphs': 0, 'dispVar_element': [], 'dispVar_names': [], 'dispVar_plot': []},
                    'map': {'nmapOutVars': 0, 'mapOutON_OFF': 0, 'mapOutVar_names': []},
                    'nhru': {'nhruOutVars': 0, 'nhruOutON_OFF': 0, 'nhruOutVar_names': []},
                    'statVar': {'nstatVars': 0, 'statsON_OFF': 0, 'statVar_element': [], 'statVar_names': []}}

        for kk, vv in grp_name[grpname].iteritems():
            self.replace_values(kk, vv)


    def get_var(self, varname):
        # Return the given variable
        if not self.__isloaded:
            self.load_file(self.__filename)

        if varname in self.__controldict:
                return self.__controldict[varname]
        return None

    def get_values(self, varname):
        """Return value(s) for given variable name"""
        if not self.__isloaded:
            self.load_file(self.__filename)

        thevar = self.get_var(varname)['values']

        if thevar is not None:
            if len(thevar) == 1:
                return thevar[0]
            else:
                return thevar
        else:
            return None

    def replace_values(self, varname, newvals):
        if not self.__isloaded:
            self.load_file(self.__filename)

        thevar = self.get_var(varname)

        if isinstance(newvals, list):
            pass
        else:
            # Convert newvals to a list
            newvals = [newvals]

        thevar['values'] = newvals

        if varname in ['statVar_names', 'statVar_element'] and len(thevar['values']) != self.get_values('nstatVars'):
            # update nstatvars
            self.replace_values('nstatVars', len(newvals))

        # Check if size of newvals array matches the oldvals array
        # if len(newvals) == len(thevar['values']):
        #     # Size of arrays match so replace the oldvals with the newvals
        #     thevar['values'] = newvals
        # else:
        #     print "ERROR: Size of oldval array and size of newval array don't match"


    def write_control_file(self, filename):
        # Write the parameters out to a file
        if not self.__isloaded:
            self.load_file(self.__filename)

        outfile = open(filename, 'w')

        for hh in self.__header:
            # Write out any header stuff
            outfile.write('%s\n' % hh)

        # Now write out the Parameter category
        #order = ['name', 'dimnames', 'valuetype', 'values']
        order = ['valuetype', 'values']

        for kk in ctl_order:
            try:
                vv = self.__controldict[kk]
            except:
                continue

        # for kk, vv in self.__controldict.iteritems():
            valnum = len(vv['values'])
            valtype = vv['valuetype']

            # Set a format string based on the valtype
            if valtype == 1:
                fmt = '%d\n'
            elif valtype == 2:
                fmt = '%f\n'
            else:
                fmt = '%s\n'

            # Write the self.__rowdelim before the variable name
            outfile.write('%s\n' % self.__rowdelim)
            outfile.write('%s\n' % kk)

            for item in order:
                # Write each variable out separated by self.__rowdelim
                val = vv[item]

                if item == 'valuetype':
                    # valnum (which is computed) must be written before valuetype
                    outfile.write('%d\n' % valnum)
                    outfile.write('%d\n' % val)
                elif item == 'values':
                    # Write one value per line
                    for xx in val:
                        outfile.write(fmt % xx)

        outfile.close()
# ***** END class control()




class streamflow(object):
    # Library to work with PRMS streamflow data files that were generated by
    #   class gov.usgs.trinli.ft.point.writer.PrmsWriter

    def __init__(self, filename, missing=-999.0, verbose=False):
        # 1) open file
        # 2) get the metaheaders
        #    get the number of header lines
        # 3) get the station list
        # 4)

        self.__missing = missing
        self.filename = filename
        self.__verbose = verbose
        self.__timecols = 6         # number columns for time in the file
        self.__headercount = None
        self.__metaheader = None
        self.__types = None
        self.__units = []
        self.__stations = None
        self.__rawdata = None
        self.__selectedStations = None
        self.__isloaded = False

        self.load_file(self.filename)



    def load_file(self, filename):

        self.__selectedStations = None  # Clear out any selected stations
        self.__metaheader = []  # Hols the column names
        self.__types = {}   # dictionary of 'Type' field in order of occurrence
        self.__units = []   # list of units in file
        self.__stations = []       # list of gage stations
        self.__stationIndex = {}    # Lookup of station id to header info

        headerNext = False
        stationNext = False

        infile = open(filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        self.__headercount = 0
        # We assume if 'ID' and 'Type' header names exist then we have a valid
        # meta-header.
        for line in it:
            self.__headercount += 1

            #print line[0:10]
            # Skip through lines until we hit the following
            if line[0:10] == '// Station':
                # Read the next line in - this is the header info
                self.__headercount += 1
                self.__metaheader = re.findall(r"[\w]+", next(it))
                break

        cnt = 0
        order = 0   # defines the order of the data types in the dataset
        curr_fcnt = 0
        st = 0

        #print '-'*10,'metaheader','-'*10
        #print self.__metaheader

        for line in it:
            self.__headercount += 1
            if line[0:10] == '//////////':
                break

            # Read in station information
            # Include question mark in regex below as a valid character since the obs
            # file uses it for missing data.
            words = re.findall(r"[\w.-]+|[?]", line)  # Break the row up
            curr_fcnt = len(words)

            # Check that number of fields remains constant
            if curr_fcnt != len(self.__metaheader):
                if self.__verbose:
                    print "WARNING: number of header fields changed from %d to %d" % (len(self.__metaheader), curr_fcnt),
                    print "\t", words
                #exit()

            if words[self.__metaheader.index('Type')] not in self.__types:
                st = cnt    # last cnt becomes the starting column of the next type
                order += 1

            # Information stored in __types array:
            # 1) Order that type was added in
            # 2) Starting index for data section
            # 3) Ending index for data section
            self.__types[words[self.__metaheader.index('Type')]] = [order,st,cnt]

            self.__stations.append(words)
            self.__stationIndex[words[0]] = cnt
            cnt += 1

        #print self.__types

        # Now read in units and add to each type
        #print '-'*10,'UNITS','-'*10
        #print line
        unittmp = next(it).split(':')[1].split(',')
        for xx in unittmp:
            unit_pair = xx.split('=')
            #print 'unit_pair:', unit_pair[0].strip(), '/', unit_pair[1].strip()
            #self.__units[unit_pair[0].strip()] = unit_pair[1].strip()
            self.__units.append([unit_pair[0].strip(), unit_pair[1].strip()])

        #print self.__units

        # Skip to the data section
        for line in it:
            self.__headercount += 1
            if line[0:10] == '##########':
                self.__headercount += 1 # plus one for good measure
                break

        #print 'headercount:', self.__headercount
        # Data section

        # The first 6 columns are [year month day hour minute seconds]
        thecols = ['year', 'month', 'day', 'hour', 'min', 'sec']

        # Add the remaining columns to the list
        for xx in self.__stations:
            thecols.append(xx[0])

        #print 'thecols:', thecols

        # Use pandas to read the data in from the remainder of the file
        # We use a custom date parser to convert the date information to a datetime
        self.__rawdata = pd.read_csv(self.filename, skiprows=self.__headercount, sep=r"\s+",
                                     header=None, names=thecols,
                                     parse_dates={'thedate': ['year', 'month', 'day', 'hour', 'min', 'sec']},
                                     date_parser=dparse, index_col='thedate')

        # Convert the missing data (-999.0) to NaNs
        self.__rawdata.replace(to_replace=self.__missing, value=np.nan, inplace=True)

        #print self.__rawdata.head()

        self.__isloaded = True


    @property
    def headercount(self):
        # Description: Returns the line number where the data begins in the given
        #              filename.
        #        Date: 2013-07-01
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__headercount


    @property
    def metaheader(self):
        # Description: Reads the "meta" header from the prms
        #        Date: 2013-06-25

        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__metaheader

    @property
    def stations(self):
        # Description: Get the list of stations in the prms file.
        #        Note: Modified to return a list of all fields for each station

        # The order of the 'Type' field dictates the gross order of the following
        # data section.  For a given 'Type' the order of the 'ID' (stations) dictates
        # order of the data.

        # Get the meta-headers for the file
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__stations


    @property
    def timecolcnt(self):
        return self.__timecols


    @property
    def types(self):
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.__types


    @property
    def data(self):
        if not self.__isloaded:
            self.load_file(self.filename)

        if self.__selectedStations is None:
            return self.__rawdata
        else:
            return self.__rawdata.ix[:, self.__selectedStations]

    @property
    def date_range(self):
        if not self.__isloaded:
            self.load_file(self.filename)

        # Return the first and last available date for valid streamflow data
        # 2015-05-19: This currently assumes it is returning a single streamgage
        tmpdf = self.data.dropna(axis=0, how='any')
        first_date = tmpdf[tmpdf.notnull()].index.min()
        last_date = tmpdf[tmpdf.notnull()].index.max()

        return (first_date, last_date)

    @property
    def numdays(self):
        """The period of record in days"""
        if not self.__isloaded:
            self.load_file(self.filename)
        return self.data.shape[0]


    @property
    def timedata(self):
        """Returns an array of time information"""
        # FIXME: This needs to be updated (2015-02-03)
        return self.data[:, 0:self.timecolcnt].astype(int)


    @property
    def units(self):
        return self.__units

    def getDataByType(self, thetype):
        """Returns data selected type (e.g. runoff)"""

        if thetype in self.__types:
            #print "Selected type '%s':" % (thetype), self.__types[thetype]
            st = self.__types[thetype][1]
            en = self.__types[thetype][2]
            #print "From %d to %d" % (st, en)
            b = self.data.iloc[:,st:en+1]

            return b
        else:
            print "not found"


    def getStationsByType(self, thetype):
        """Returns station IDs for a given type (e.g. runoff)"""

        if thetype in self.__types:
            #print "Selected type '%s':" % (thetype), self.__types[thetype]
            st = self.__types[thetype][1]
            en = self.__types[thetype][2]
            #print "From %d to %d" % (st, en)
            b = self.stations[st:en+1]

            return b
        else:
            print "not found"


    def selectByStation(self, streamgages):
        """Selects one or more streamgages from the dataset"""
        # The routine writeSelected() will write selected streamgages and data
        # to a new PRMS streamflow file.
        # Use clearSelectedStations() to clear any current selection.
        if isinstance(streamgages, list):
            self.__selectedStations = streamgages
        else:
            self.__selectedStations = [streamgages]



    def clearSelectedStations(self):
        """Clears any selected streamgages"""
        self.__selectedStations = None


    def writeSelectedStations(self, filename):
        """Writes station observations to a new file"""
        # Either writes out all station observations or, if stations are selected,
        # then a subset of station observations.

        # Sample header format

        # $Id:$
        # ////////////////////////////////////////////////////////////
        # // Station metadata (listed in the same order as the data):
        # // ID    Type Latitude Longitude Elevation
        # // <station info>
        # ////////////////////////////////////////////////////////////
        # // Unit: runoff = ft3 per sec, elevation = feet
        # ////////////////////////////////////////////////////////////
        # runoff <number of stations for each type>
        # ################################################################################

        topLine = '$Id:$\n'
        sectionSep = '////////////////////////////////////////////////////////////\n'
        metaHeader1 = '// Station metadata (listed in the same order as the data):\n'
        #metaHeader2 = '// ID    Type Latitude Longitude Elevation'
        metaHeader2 = '// %s\n' % ' '.join(self.metaheader)
        dataSection = '################################################################################\n'

        # ----------------------------------
        # Get the station information for each selected station
        typeCount = {}  # Counts the number of stations for each type of data (e.g. 'runoff')
        stninfo = ''
        if self.__selectedStations is None:
            for xx in self.__stations:
                if xx[1] not in typeCount:
                    # index 1 should be the type field
                    typeCount[xx[1]] = 0
                typeCount[xx[1]] += 1

                stninfo += '// %s\n' % ' '.join(xx)
        else:
            for xx in self.__selectedStations:
                cstn = self.__stations[self.__stationIndex[xx]]

                if cstn[1] not in typeCount:
                    # index 1 should be the type field
                    typeCount[cstn[1]] = 0

                typeCount[cstn[1]] += 1

                stninfo += '// %s\n' % ' '.join(cstn)
        #stninfo = stninfo.rstrip('\n')

        # ----------------------------------
        # Get the units information
        unitLine = '// Unit:'
        for uu in self.__units:
            unitLine += ' %s,' % ' = '.join(uu)
        unitLine = '%s\n' % unitLine.rstrip(',')

        # ----------------------------------
        # Create the list of types of data that are being included
        tmpl = []

        # Create list of types in the correct order
        for kk,vv in self.__types.iteritems():
            if kk in typeCount:
                tmpl.insert(vv[0], [kk, typeCount[kk]])

        typeLine = ''
        for tt in tmpl:
            typeLine += '%s %d\n' % (tt[0], tt[1])
        #typeLine = typeLine.rstrip('\n')

        # Write out the header to the new file
        outfile = open(filename, 'w')
        outfile.write(topLine)
        outfile.write(sectionSep)
        outfile.write(metaHeader1)
        outfile.write(metaHeader2)
        outfile.write(stninfo)
        outfile.write(sectionSep)
        outfile.write(unitLine)
        outfile.write(sectionSep)
        outfile.write(typeLine)
        outfile.write(dataSection)

        # Write out the data to the new file
        # Using quoting=csv.QUOTE_NONE results in an error when using a customized  date_format
        # A kludgy work around is to write with quoting and then re-open the file
        # and write it back out, stripping the quote characters.
        self.data.to_csv(outfile, index=True, header=False, date_format='%Y %m %d %H %M %S', sep=' ')
        outfile.close()

        old = open(filename,'r').read()
        new = re.sub('["]', '', old)
        open(filename, 'w').write(new)



    # def getRecurrenceInterval(self, thetype):
    #     """Returns the recurrence intervals for each station"""
    #
    #     # Copy the subset of data
    #     xx = self.seldata(thetype)
    #
    #     ri = np.zeros(xx.shape)
    #     ri[:,:] = -1.
    #
    #     # for each station we need to compute the RI for non-zero values
    #     for ss in range(0,xx.shape[1]):
    #         tmp = xx[:,ss]              # copy values for current station
    #
    #         # Get array of indices that would result in a sorted array
    #         sorted_ind = np.argsort(tmp)
    #         #print "sorted_ind.shape:", sorted_ind.shape
    #
    #         numobs = tmp[(tmp > 0.0),].shape[0]  # Number of observations > 0.
    #         nyr = float(numobs / 365)     # Number of years of non-zero observations
    #
    #         nz_cnt = 0  # non-zero value counter
    #         for si in sorted_ind:
    #             if tmp[si] > 0.:
    #                 nz_cnt += 1
    #                 rank = numobs - nz_cnt + 1
    #                 ri[si,ss] = (nyr + 1.) / float(rank)
    #                 #print "%s: [%d]: %d %d %0.3f %0.3f" % (ss, si,  numobs, rank, tmp[si], ri[si,ss])
    #
    #     return ri

# ***** END of class streamflow()




class parameters(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2015-01-29
    # Description: Class object to handle reading and writing the PRMS
    #              parameter files which have been generated by Java.

    def __init__(self, filename):
        # 1) open file
        # 2) read file contents

        self.__isloaded = False
        self.__filename = filename
        self.__paramdict = {}
        self.__header = []
        self.__catdelim = '**'      # Delimiter for categories of variables
        self.__rowdelim = '####'    # Used to delimit variables
        self.__valtypes = ['', 'integer', 'float', 'double', 'string']

        self.load_file()
    # END __init__

    def __getattr__(self, item):
        # Undefined attributes will look up the given parameter
        return self.get_var(item)

    @property
    def filename(self):
        # if not self.__isloaded:
        #     self.load_file(self.__filename)
        return self.__filename

    @filename.setter
    def filename(self, fname):
        self.__isloaded = False

        self.__paramdict = {}
        self.__header = []

        self.__filename = fname

        self.load_file()

    @property
    def headers(self):
        """Returns the headers read from the parameter file"""
        return self.__header


    @property
    def vars(self):
        """Return a structure of loaded variables"""
        if not self.__isloaded:
            self.load_file()

        varlist = []
        parent = self.__paramdict['Parameters']

        for ee in parent:
            varlist.append(ee['name'])

        return varlist

    @property
    def dimensions(self):
        """Return a list of dimensions"""
        if not self.__isloaded:
            self.load_file()

        # dimlist = []
        # parent = self.__paramdict['Dimensions']
        #
        # for kk,vv in parent.iteritems():
        #     dimlist.append((kk, vv))
        #
        # return dimlist
        return self.__paramdict['Dimensions']


    def vars_integrity(self):
        """Check all parameter variables for proper array size"""

        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Parameters']

        for ee in parent:
            print '%20s' % ee['name'],

            total_size = 1
            for dd in ee['dimnames']:
                total_size *= self.get_dim(dd)

            if ee['values'].size == total_size:
                print 'OK'
            else:
                print 'BAD'


    def load_file(self):
        # Read the parameter file into memory and parse it

        self.__paramdict = {}   # Initialize the parameter dictionary
        self.__header = []      # Initialize the list of file headers

        infile = open(self.__filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        for line in it:
            dupskip = False

            if line[0:4] == '$Id:' or line[0:7] == 'Version' or \
                    line[0:17] == 'Default Parameter' or line[0:15] == 'PRMS version 4':
                # TODO: The test for header information is clunky and brittle. Need to fix.
                self.__header.append(line)
                continue
            if line[0:2] == self.__catdelim:
                if line == '** Dimensions **':
                    # new category
                    #print 'Adding Dimensions category'
                    self.__paramdict['Dimensions'] = {}
                    dimsection = True
                elif line == '** Parameters **':
                    #print 'Adding Parameters category'
                    self.__paramdict['Parameters'] = []
                    dimsection = False
            elif line == self.__rowdelim:
                continue
            else:
                # We're within a category section and dealing with variables
                if dimsection:
                    # -------------------------------------------------------
                    # DIMENSIONS section
                    # -------------------------------------------------------
                    # The 'Dimensions' section only has scalar integer variables
                    self.__paramdict['Dimensions'][line] = int(next(it))
                else:
                    # -------------------------------------------------------
                    # PARAMETER section
                    # -------------------------------------------------------
                    vardict = {}    # temporary to build variable info
                    varname = line.split(' ')[0]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check for duplicate variable name
                    for kk in self.__paramdict['Parameters']:
                        # Check for duplicate variables (that couldn't happen! :))
                        # If it does skip to the next variable in the parameter file
                        if varname == kk['name']:
                            print '%s: Duplicate parameter name.. skipping' \
                                  % varname
                            dupskip = True
                            break

                    if dupskip:
                        try:
                            while next(it) != self.__rowdelim:
                                pass
                        except StopIteration:
                            # We hit the end of the file
                            continue
                        continue
                    # END check for duplicate varnames
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    vardict['name'] = varname

                    # Read in the dimension names
                    numdim = int(next(it))    # number of dimensions for this variable
                    vardict['dimnames'] = [next(it) for dd in xrange(numdim)]

                    # Lookup dimension size for each dimension name
                    arr_shp = [self.__paramdict['Dimensions'][dd] for dd in vardict['dimnames']]

                    numval = int(next(it))  # Denotes the number of data values we have. Should match dimensions.
                    valuetype = int(next(it))   # Datatype of the values
                    vardict['valuetype'] = int(valuetype)

                    try:
                        # Read in the data values
                        vals = []

                        while True:
                            cval = next(it)

                            if cval == self.__rowdelim or cval.strip() == '':
                                break
                            vals.append(cval)
                    except StopIteration:
                        # Hit the end of the file
                        continue

                    if len(vals) != numval:
                        print '%s: number of values does not match dimension size (%d != %d).. skipping' \
                            % (varname, len(vals), numval)
                    else:
                        # Convert the values to the correct datatype
                        # 20151118 PAN: found a value of 1e+05 in nhm_id for r17 caused this to fail
                        #               even though manaully converting the value to int works.
                        try:
                            if valuetype == 1:      # integer
                                vals = [int(vals) for vals in vals]
                            elif valuetype == 2:    # float
                                vals = [float(vals) for vals in vals]
                        except ValueError:
                            print "%s: value type and defined type (%s) don't match" \
                                  % (varname, self.__valtypes[valuetype])

                        # Add to dictionary as a numpy array
                        vardict['values'] = np.array(vals).reshape(arr_shp)
                        self.__paramdict['Parameters'].append(vardict)
        self.__isloaded = True
    # END **** read_params()

    def get_var(self, varname):
        # Return the given variable

        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Parameters']

        for ee in parent:
            if ee['name'] == varname:
                return ee
        return None

    def get_dim(self, dimname):
        # Return the size of the specified dimension

        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Dimensions']

        if dimname in parent:
            return parent[dimname]
        return None


    def add_param(self, name, dimnames, valuetype, values):
        # Add a new parameter
        if not self.__isloaded:
            self.load_file()

        # Check that valuetype is valid
        if valuetype not in [1, 2, 3, 4]:
            print "ERROR: Invalide valuetype was specified"
            return

        # Check that total dimension size matches number of values supplied
        if isinstance(dimnames, list):
            # multiple dimensions supplied
            tsize = 1

            for dd in dimnames:
                tsize *= self.get_dim(dd)

            if tsize != len(values):
                print "ERROR: Number of values (%d) does not match size of dimensions (%d)" % (len(values), tsize)
                return
        else:
            # single dimension
            tsize = self.get_dim(dimnames)

            if isinstance(values, list):
                print "ERROR: Scalar dimensions specified but of list of values given"
                return

        parent = self.__paramdict['Parameters']

        # Make sure the parameter doesn't already exist
        for ee in parent:
            if ee['name'] == name:
                print 'ERROR: Parameter name already exists, use replace_values() instead.'
                return

        if isinstance(dimnames, list):
            parent.append({'name': name, 'dimnames': dimnames, 'valuetype': valuetype, 'values': values})
        else:
            parent.append({'name': name, 'dimnames': [dimnames], 'valuetype': valuetype, 'values': values})


    def check_var(self, varname):
        # Check a variable to see if the number of values it has is
        # consistent with the given dimensions

        if not self.__isloaded:
            self.load_file()

        thevar = self.get_var(varname)

        # Get the defined size for each dimension used by the variable
        total_size = 1
        for dd in thevar['dimnames']:
            total_size *= self.get_dim(dd)

        if thevar['values'].size == total_size:
            # The number of values for the defined dimensions match
            print 'OK'
        else:
            print 'BAD'

    def copy_param(self, varname, filename):
        """Copies selected varname from given src input parameter file (filename).
        The incoming parameter is verified to have the same dimensions and sizes as
        the destination."""

        # TODO: Expand this handle either a single varname or a list of varnames
        srcparamfile = parameters(filename)
        srcparam = srcparamfile.get_var(varname)
        self.add_param(srcparam['name'], srcparam['dimnames'], srcparam['valuetype'], srcparam['values'])
        del(srcparamfile)


    def replace_values(self, varname, newvals, newdims=None):
        """Replaces all values for a given variable/parameter. Size of old and new arrays/values must match."""
        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Parameters']
        thevar = self.get_var(varname)

        if newdims is None:
            # We are not changing dimensions of the variable/parameter, just the values
            # Check if size of newvals array matches the oldvals array
            if isinstance(newvals, list) and len(newvals) == thevar['values'].size:
                # Size of arrays match so replace the oldvals with the newvals
                thevar['values'][:] = newvals
            elif isinstance(newvals, np.ndarray) and newvals.size == thevar['values'].size:
                # newvals is a numpy ndarray
                # Size of arrays match so replace the oldvals with the newvals
                thevar['values'][:] = newvals
            elif thevar['values'].size == 1:
                # This is a scalar value
                if isinstance(newvals, float):
                    thevar['values'] = [newvals]
                elif isinstance(newvals, int):
                    thevar['values'] = [newvals]
            else:
                print "ERROR: Size of oldval array and size of newval array don't match"
        else:
            # The dimensions are being changed and new values provided
            pass


    def resize_dim(self, dimname, newsize):
        """Changes the size of the given dimension.
           This does *not* check validity of parameters that use the dimension.
           Check variable integrity before writing parameter file."""

        # Some dimensions are related to each other.
        related_dims = {'ndepl': 'ndeplval', 'nhru': ['nssr', 'ngw'],
                        'nssr': ['nhru', 'ngw'], 'ngw': ['nhru', 'nssr']}

        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Dimensions']

        if dimname in parent:
            parent[dimname] = newsize

            # Also update related dimensions
            if dimname in related_dims:
                if dimname == 'ndepl':
                    parent[related_dims[dimname]] = parent[dimname] * 11
                elif dimname in ['nhru', 'nssr', 'ngw']:
                    for dd in related_dims[dimname]:
                        parent[dd] = parent[dimname]
            return True
        else:
            return False


    def update_values_by_hru(self, varname, newvals, hru_index):
        """Updates parameter/variable with new values for a a given HRU.
           This is used when merging data from an individual HRU into a region"""
        if not self.__isloaded:
            self.load_file()

        parent = self.__paramdict['Parameters']
        thevar = self.get_var(varname)

        if len(newvals) == 1:
            thevar['values'][(hru_index-1)] = newvals
        elif len(newvals) == 2:
            thevar['values'][(hru_index-1), :] = newvals
        elif len(newvals) == 3:
            thevar['values'][(hru_index-1), :, :] = newvals


    def distribute_mean_value(self, varname, new_mean):
        #def redistribute_mean(old_vals, new_mean):
        # Redistribute mean value to set of multiple initial values
        # see Hay and Umemoto, 2006 (p. 11)

        old_vals = self.get_var(varname)['values']
        if len(old_vals) > 1:
            # This parameter is a list of values
            ZC = 10.    # Constant to avoid zero values
            new_vals = []

            old_mean = sum(old_vals) / float(len(old_vals))

            for vv in old_vals:
                new_vals.append((((new_mean + ZC) * (vv + ZC)) / (old_mean + ZC)) - ZC)

            self.replace_values(varname, new_vals)
        else:
            self.replace_values(varname, new_mean)


    def pull_hru2(self, hru_index, filename):
        # Pulls a single HRU out by index and writes a new parameter file for that HRU
        # This version greatly improves on the original pull_hru by just reading from the parameter
        # data structure and writing the modify file directly instead of modifying a copy of the
        # original parameter structure and then writing it out.

        split_dims = ['nhru', 'nssr', 'ngw']
        ndepl = self.get_dim('ndepl')
        nhru = self.get_dim('nhru')
        order = ['name', 'dimnames', 'valuetype', 'values']

        # Parameters that need parent information saved
        parent_info = {'hru_segment': 'parent_hru',
                       'tosegment': 'parent_segment'}

        #segvars = ['K_coef', 'obsin_segment', 'tosegment', 'x_coef', 'segment_type', 'segment_flow_init', 'parent_segment']
        segvars = ['K_coef', 'obsin_segment', 'x_coef', 'segment_type', 'segment_flow_init', 'parent_segment']

        # Adjustment values for select dimensions
        dim_adj = {'nobs': 1, 'nsegment':1, 'npoigages':1, }

        # ===================================================================
        outfile = open(filename, 'w')

        # -------------------------------------------------------------------
        for hh in self.__header:
            # Write out any header stuff
            outfile.write('%s\n' % hh)

        # Dimension section must be written first
        dimparent = self.__paramdict['Dimensions']

        outfile.write('%s Dimensions %s\n' % (self.__catdelim, self.__catdelim))

        # -------------------------------------------------------------------
        for kk, vv in dimparent.iteritems():
            # Write the dimension names and values separated by self.__rowdelim
            outfile.write('%s\n' % self.__rowdelim)
            outfile.write('%s\n' % kk)

            # Adjust the split_dims e.g. nhru, ngw, nssr and asssorted other dimensions
            if kk in split_dims:
                outfile.write('%d\n' % 1)
            elif kk in dim_adj:
                outfile.write('%d\n' % dim_adj[kk])
            elif kk == 'ndepl' and ndepl == nhru:
                outfile.write('%d\n' % 1)
            elif kk == 'ndeplval' and ndepl == nhru:
                outfile.write('%d\n' % 11)
            else:
                outfile.write('%d\n' % vv)

        # Now write out the Parameter category
        paramparent = self.__paramdict['Parameters']

        outfile.write('%s Parameters %s\n' % (self.__catdelim, self.__catdelim))

        # Get the segment index for the HRU we are grabbing
        seg_idx = self.get_var('hru_segment')['values'][hru_index]
        sys.stdout.write('\rHRU %06d to segment: %06d' % (hru_index, seg_idx))
        sys.stdout.flush()

        # -------------------------------------------------------------------
        for vv in paramparent:
            valtype = vv['valuetype']

            # Set a format string based on the valtype
            if valtype == 1:
                fmt = '%s\n'
            elif valtype == 2:
                #fmt = '%0.8f\n'
                fmt = '%s\n'
            else:
                fmt = '%s\n'

            if bool(set(vv['dimnames']).intersection(split_dims)):
                # dealing with nhru, nssr, or ngw

                if len(vv['dimnames']) == 2:
                    the_values = vv['values'][hru_index,:]
                    dimsize = vv['values'][hru_index,:].size
                elif len(vv['dimnames']) == 1:
                    the_values = vv['values'][hru_index]
                    dimsize = vv['values'][hru_index].size
            else:
                dimsize = vv['values'].size
                the_values = vv['values']

            # Special overrides for some parameters
            if vv['name'] in segvars:
                the_values = np.array([vv['values'][seg_idx-1]])
                dimsize = 1
            elif vv['name'] == 'hru_segment':
                the_values = np.array([1])
                dimsize = 1
            elif vv['name'] in ['tosegment', 'poi_gage_segment', 'poi_type', 'poi_gage_id']:
                the_values = np.array([0])
                dimsize = 1

            for item in order:
                # Write each variable write out separated by self.__rowdelim
                val = vv[item]

                if item == 'dimnames':
                    # Write number of dimensions first
                    outfile.write('%d\n' % len(val))
                    for dd in val:
                        # Write dimension names
                        outfile.write('%s\n' % dd)
                elif item == 'valuetype':
                    # dimsize (which is computed) must be written before valuetype
                    outfile.write('%d\n' % dimsize)
                    outfile.write('%d\n' % val)
                elif item == 'values':
                    # Write one value per line
                    for xx in the_values.flatten():
                        outfile.write(fmt % xx)
                elif item == 'name':
                    # Write the self.__rowdelim before the variable name
                    outfile.write('%s\n' % self.__rowdelim)
                    outfile.write('%s\n' % val)

        # Not quite done... add parent information so we can stitch the HRUs back together later
        # order = ['name', 'dimnames', 'valuetype', 'values']

        # Write the parent_segment information
        # The parent_segment is the value that was used in the parent parameter file for hru_segment at
        # the given parent_hru index. It shouldn't be needed for checking a single HRU back into the
        # parent parameter file.
        outfile.write('%s\n' % self.__rowdelim)
        outfile.write('%s\n' % 'parent_segment')
        outfile.write('%d\n' % 1)
        outfile.write('%s\n' % 'nsegment')
        outfile.write('%d\n' % 1)
        outfile.write('%d\n' % 1)
        outfile.write('%d\n' % seg_idx)

        # Write the parent_hru information
        # The parent_hru is the index to use when checking a single HRU file back into the parent parameter file
        outfile.write('%s\n' % self.__rowdelim)
        outfile.write('%s\n' % 'parent_hru')
        outfile.write('%d\n' % 1)
        outfile.write('%s\n' % 'nhru')
        outfile.write('%d\n' % 1)
        outfile.write('%d\n' % 1)
        outfile.write('%d\n' % (hru_index+1))

        outfile.close()


    def write_select_param_file(self, filename, selection):
        # Write selected subset of parameters to a new parameter file
        if not self.__isloaded:
            self.load_file()

        outfile = open(filename, 'w')

        # Write out the Parameter category
        order = ['name', 'dimnames', 'valuetype', 'values']

        for ss in selection:
            vv = self.get_var(ss)
            dimsize = vv['values'].size
            valtype = vv['valuetype']

            # Set a format string based on the valtype
            if valtype == 1:
                fmt = '%s\n'
            elif valtype == 2:
                #fmt = '%f\n'
                fmt = '%s\n'
            else:
                fmt = '%s\n'

            for item in order:
                # Write each variable out separated by self.__rowdelim
                val = vv[item]

                if item == 'dimnames':
                    # Write number of dimensions first
                    outfile.write('%d\n' % len(val))
                    for dd in val:
                        # Write dimension names
                        outfile.write('%s\n' % dd)
                elif item == 'valuetype':
                    # dimsize (which is computed) must be written before valuetype
                    outfile.write('%d\n' % dimsize)
                    outfile.write('%d\n' % val)
                elif item == 'values':
                    # Write one value per line
                    for xx in val.flatten():
                        outfile.write(fmt % xx)
                elif item == 'name':
                    # Write the self.__rowdelim before the variable name
                    outfile.write('%s\n' % self.__rowdelim)
                    outfile.write('%s 10\n' % val)

        outfile.close()


    def write_param_file(self, filename):
        # Write the parameters out to a file

        if not self.__isloaded:
            self.load_file()

        outfile = open(filename, 'w')

        for hh in self.__header:
            # Write out any header stuff
            outfile.write('%s\n' % hh)

        # Dimension section must be written first
        dimparent = self.__paramdict['Dimensions']

        outfile.write('%s Dimensions %s\n' % (self.__catdelim, self.__catdelim))

        for kk, vv in dimparent.iteritems():
            # Write the dimension names and values separated by self.__rowdelim
            outfile.write('%s\n' % self.__rowdelim)
            outfile.write('%s\n' % kk)
            outfile.write('%d\n' % vv)

        # Now write out the Parameter category
        paramparent = self.__paramdict['Parameters']
        order = ['name', 'dimnames', 'valuetype', 'values']

        outfile.write('%s Parameters %s\n' % (self.__catdelim, self.__catdelim))

        for vv in paramparent:
            dimsize = vv['values'].size
            valtype = vv['valuetype']

            # Set a format string based on the valtype
            if valtype == 1:
                fmt = '%s\n'
            elif valtype == 2:
                #fmt = '%f\n'
                fmt = '%s\n'
            else:
                fmt = '%s\n'

            for item in order:
                # Write each variable out separated by self.__rowdelim
                val = vv[item]

                if item == 'dimnames':
                    # Write number of dimensions first
                    outfile.write('%d\n' % len(val))
                    for dd in val:
                        # Write dimension names
                        outfile.write('%s\n' % dd)
                elif item == 'valuetype':
                    # dimsize (which is computed) must be written before valuetype
                    outfile.write('%d\n' % dimsize)
                    outfile.write('%d\n' % val)
                elif item == 'values':
                    # Write one value per line
                    for xx in val.flatten():
                        outfile.write(fmt % xx)
                elif item == 'name':
                    # Write the self.__rowdelim before the variable name
                    outfile.write('%s\n' % self.__rowdelim)
                    outfile.write('%s\n' % val)

        outfile.close()
# ***** END of class parameters()




class statvar(object):
    def __init__(self, filename=None, missing=-999.0):
        self.__timecols = 6         # number columns for time in the file
        self.__missing = missing    # what is considered a missing value?
        self.filename = filename    # trigger the filename setter

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
        #print "Number of variables: %d" % (numvars)

        # The next numvar rows contain a variable name followed by a number which
        # indicates the number of columns used by that variable.
        # The relative order of the variable in the list indicates the column
        # the variable data is found in.
        self.__isloaded = False
        self.__vars = {}
        self.__header = []

        # The first 7 columns are [record year month day hour minute seconds]
        self.__header = ['rec', 'year', 'month', 'day', 'hour', 'min', 'sec']

        for rr in range(0,numvars):
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
                    self.__header.append('%s_%d' % (varname, dd+1))
                else:
                    self.__header.append('%s' % varname)

        # Now load the data

        # Use pandas to read the data in from the remainder of the file
        # We use a custom date parser to convert the date information to a datetime
        self.__rawdata = pd.read_csv(infile, sep=r"\s+", header=None, names=self.__header,
                                     parse_dates={'thedate': ['year', 'month', 'day', 'hour', 'min', 'sec']},
                                     date_parser=dparse, index_col='thedate')
        
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



