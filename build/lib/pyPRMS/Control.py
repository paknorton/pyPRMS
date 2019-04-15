#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

from pyPRMS.constants import ctl_order, ctl_module_params


class Control(object):
    """
    Class which handles the processing of PRMS control files.
    """
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2015-02-05
    # Description: Class object to handle reading and writing the PRMS
    #              control files.

    def __init__(self, filename):
        # 1) open file
        # 2) read file contents

        self.__isloaded = False
        self.__filename = filename
        self.__controldict = {}
        self.__modules = {}  # Initialize dictionary of selected module names
        self.__header = []
        self.__rowdelim = '####'  # Used to delimit variables
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
                    outstr += '{0:s}: {1:s}, {2:s}\n'.format(xx, self.__valtypes[pp['valuetype']], str(pp['values'][0]))
                else:
                    outstr += '{0:s}: {1:s}, {2:d} values\n'.format(xx, self.__valtypes[pp['valuetype']],
                                                                    len(pp['values']))
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
    def modules(self):
        return self.__modules

    @property
    def vars(self):
        # Return a list of variables
        if not self.__isloaded:
            self.load_file(self.__filename)

        varlist = []

        for cc in self.__controldict:
            varlist.append(cc)
        return varlist

    @property
    def rawvars(self):
        return self.__controldict

    def add(self, varname, vartype, val):
        """Add a variable to the control file.

        Args:
            varname: The variable name to use.
            vartype: The datatype of the variable (one of 'integer', 'string', 'double', 'float').
            val: The value to assign to the variable
        """
        # Add a variable to the control file
        if not self.__isloaded:
            self.load_file(self.__filename)

        cvars = self.vars

        if varname in cvars:
            print("ERROR: %s already exists, use replace_values() instead")
            return

        if not isinstance(val, list):
            val = [val]

        self.__controldict[varname] = {'valuetype': vartype, 'values': val}

    def add_missing(self):
        # Add missing control file variables
        pass

    def load_file(self, filename):
        """Load a control file.

        Reads the contents of a control file into the class.

        Args:
            filename: The name of the control file to read.
        """
        # Read the control file into memory and parse it
        self.__isloaded = False
        self.__modules = {}  # Initialize dictionary of selected module names
        self.__controldict = {}  # Initialize the control dictionary

        infile = open(filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        for fidx, line in enumerate(it):
            if fidx == 0:
                self.__header.append(line)
                continue
            elif line == self.__rowdelim:
                continue
            else:
                # We're dealing with a control parameter/variable
                # We're in a parameter section
                vardict = {}  # temporary to build variable info
                varname = line.split(' ')[0]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check for duplicate variable name
                if varname in self.__controldict:
                    # Check for duplicate variables (that couldn't happen! :))
                    # If it does skip to the next variable in the parameter file
                    print('Duplicate variable name, %s, in Parameters section.. skipping' % varname)

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

                numval = int(next(it))  # number of values for this variable
                valuetype = int(next(it))  # Variable type (1 - integer, 2 - float, 4 - character)
                # print '\tnumval:', numval
                # print '\tvaluetype:', valuetype

                vardict['valuetype'] = int(valuetype)
                vals = []

                for vv in range(0, numval):
                    try:
                        if valuetype == 1:  # integer
                            vals.append(int(next(it)))
                        elif valuetype == 2:  # float
                            vals.append(float(next(it)))
                        else:  # character
                            vals.append(next(it))
                    except ValueError:
                        print("varname: %s value type and defined type (%s) don't match" %
                              (varname, self.__valtypes[valuetype]))
                vardict['values'] = vals

                if len(vals) != numval:
                    print('ERROR: Not enough values provided for %s' % varname)
                    print('       Expect %d, got %d' % (numval, len(vals)))

                # Check if there are too many values specified
                try:
                    cnt = numval
                    while next(it) != self.__rowdelim:
                        cnt += 1

                    if cnt > numval:
                        print('WARNING: Too many values specified for %s' % varname)
                        print('       %d expected, %d given' % (numval, cnt))
                        print('       Keeping first %d values' % numval)
                except StopIteration:
                    # Hit the end of the file
                    pass
                    # continue
                # self.__controldict[varname].append(vardict)
                self.__controldict[varname] = vardict

                # If this is a module-related parameter then add to __modules
                if varname in ctl_module_params:
                    if len(vardict['values']) != 1:
                        print("ERROR: %s should only have a single entry" % varname)
                    else:
                        if vardict['values'][0] not in self.__modules:
                            self.__modules[vardict['values'][0]] = []
                        self.__modules[vardict['values'][0]].append(varname)
        # ***** END for line in it

        # Add modules that should always included
        def_mods = {'basin': ['basin_def'], 'soltab': ['potet_def'],
                    'intcp': ['intcp_def'], 'snowcomp': ['snow_def'],
                    'gwflow': ['gw_def'], 'soilzone': ['soil_def'],
                    'basin_sum': ['summary_def']}
        for (kk, vv) in iteritems(def_mods):
            self.__modules[kk] = vv

        self.__isloaded = True

    # END **** load_file()

    def clear_parameter_group(self, group_name):
        """Clear a parameter group.

        Given a single parameter group name will clear out values for that parameter
           and all related parameters.

        Args:
            group_name: The name of a group of related parameters.
                One of 'statVar', 'aniOut', 'mapOut', 'dispVar', 'nhruOut'.
        """

        groups = {'aniOut': {'naniOutVars': 0, 'aniOutON_OFF': 0, 'aniOutVar_names': []},
                  'dispVar': {'ndispGraphs': 0, 'dispVar_element': [], 'dispVar_names': [], 'dispVar_plot': []},
                  'mapOut': {'nmapOutVars': 0, 'mapOutON_OFF': 0, 'mapOutVar_names': []},
                  'nhruOut': {'nhruOutVars': 0, 'nhruOutON_OFF': 0, 'nhruOutVar_names': []},
                  'statVar': {'nstatVars': 0, 'statsON_OFF': 0, 'statVar_element': [], 'statVar_names': []}}

        for (kk, vv) in iteritems(groups[group_name]):
            if kk in self.__controldict:
                self.replace_values(kk, vv)

    def get_var(self, varname):
        """Get a control file variable.

        Args:
            varname: The name of the variable to retrieve.

        Returns:
            Returns a controldict entry.
        """
        # Return the given variable
        if not self.__isloaded:
            self.load_file(self.__filename)

        if varname in self.__controldict:
            return self.__controldict[varname]
        return None

    def get_values(self, varname):
        """Get values for a control file variable.

        Args:
            varname: The name of the control file variable.

        Returns:
            Returns the value(s) associated with the control file variable
        """
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
        # order = ['name', 'dimnames', 'valuetype', 'values']
        order = ['valuetype', 'values']

        # Control file parameters may change. We'll use ctl_order to insure
        # certain parameters are always ordered, but will be followed by any
        # remaining non-ordered parameters.
        curr_ctl = set(self.__controldict.keys())
        curr_order = set(ctl_order)
        unordered_set = curr_ctl.difference(curr_order)

        # Add parameters that are missing in the ordered set at the end of the list
        ctl_order.extend(list(unordered_set))

        for kk in ctl_order:
            if kk in self.__controldict:
                vv = self.__controldict[kk]
            else:
                continue

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
