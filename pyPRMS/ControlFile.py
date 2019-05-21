#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

# from collections import OrderedDict
# try:
#     import xml.etree.cElementTree as xmlET
# except ImportError:
import xml.etree.ElementTree as xmlET

import io
import pkgutil

from pyPRMS.constants import DATA_TYPES, VAR_DELIM
from pyPRMS.Control import Control
from pyPRMS.Exceptions_custom import ControlError


class ControlFile(Control):
    """
    Class which handles the processing of PRMS control files.
    """
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18
    # Description: Class object to handle reading and writing PRMS
    #              control files.

    def __init__(self, filename):
        super(ControlFile, self).__init__()
        # 1) open file
        # 2) read file contents

        self.__isloaded = False
        self.__filename = filename
        self.filename = filename

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, fname):
        self.__isloaded = False
        self.__filename = fname
        self._read()

    def _read(self):
        """Load a control file.

        Reads the contents of a control file into the class.
        """

        # Read the control file into memory and parse it
        self.__isloaded = False

        # First read control.xml from the library
        # This makes sure any missing variables from the control file
        # end up with default values
        xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/control.xml').decode('utf-8'))
        xml_tree = xmlET.parse(xml_fh)
        xml_root = xml_tree.getroot()

        for elem in xml_root.findall('control_param'):
            name = elem.attrib.get('name')
            datatype = int(elem.find('type').text)

            self.add(name)
            self.get(name).datatype = datatype

            if name in ['start_time', 'end_time']:
                # Hack to handle PRMS weird approach to dates
                dt = elem.find('default').text.split('-')
                if len(dt) < 6:
                    # pad short date with zeros for hms
                    dt.extend([0 for _ in range(6 - len(dt))])
                self.get(name).default = dt
            else:
                # print('{}: {} {}'.format(name, type(elem.find('default').text), elem.find('default').text))
                self.get(name).default = elem.find('default').text

            self.get(name).description = elem.find('desc').text

            if elem.find('force_default') is not None:
                self.get(name).force_default = elem.find('force_default').text

            outvals = {}
            for cvals in elem.findall('./values'):
                self.get(name).value_repr = cvals.attrib.get('type')

                for cv in cvals.findall('./value'):
                    outvals[cv.attrib.get('name')] = []

                    for xx in cv.text.split(','):
                        outvals[cv.attrib.get('name')].append(xx)
            self.get(name).valid_values = outvals

        header_tmp = []
        infile = open(self.filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        for fidx, line in enumerate(it):
            if fidx == 0:
                header_tmp.append(line)
                continue
            elif line == VAR_DELIM:
                continue
            else:
                # We're dealing with a control parameter/variable
                # We're in a parameter section
                varname = line.split(' ')[0]

                numval = int(next(it))  # number of values for this variable
                valuetype = int(next(it))  # Variable type (1 - integer, 2 - float, 4 - character)

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
                              (varname, DATA_TYPES[valuetype]))

                if len(vals) != numval:
                    print('ERROR: Not enough values provided for %s' % varname)
                    print('       Expect %d, got %d' % (numval, len(vals)))

                try:
                    cnt = numval
                    while next(it) != VAR_DELIM:
                        cnt += 1

                    if cnt > numval:
                        print('WARNING: Too many values specified for %s' % varname)
                        print('       %d expected, %d given' % (numval, cnt))
                        print('       Keeping first %d values' % numval)
                except StopIteration:
                    # Hit the end of the file
                    pass

                try:
                    self.add(varname)
                except ControlError:
                    pass

                self.get(varname).datatype = int(valuetype)
                self.get(varname).values = vals

        self.header = header_tmp
        self.__isloaded = True

    # def clear_parameter_group(self, group_name):
    #     """Clear a parameter group.
    #
    #     Given a single parameter group name will clear out values for that parameter
    #        and all related parameters.
    #
    #     Args:
    #         group_name: The name of a group of related parameters.
    #             One of 'statVar', 'aniOut', 'mapOut', 'dispVar', 'nhruOut'.
    #     """
    #
    #     groups = {'aniOut': {'naniOutVars': 0, 'aniOutON_OFF': 0, 'aniOutVar_names': []},
    #               'dispVar': {'ndispGraphs': 0, 'dispVar_element': [], 'dispVar_names': [], 'dispVar_plot': []},
    #               'mapOut': {'nmapOutVars': 0, 'mapOutON_OFF': 0, 'mapOutVar_names': []},
    #               'nhruOut': {'nhruOutVars': 0, 'nhruOutON_OFF': 0, 'nhruOutVar_names': []},
    #               'statVar': {'nstatVars': 0, 'statsON_OFF': 0, 'statVar_element': [], 'statVar_names': []}}
    #
    #     for (kk, vv) in iteritems(groups[group_name]):
    #         if kk in self.__controldict:
    #             self.replace_values(kk, vv)
    #
    # def get_var(self, varname):
    #     """Get a control file variable.
    #
    #     Args:
    #         varname: The name of the variable to retrieve.
    #
    #     Returns:
    #         Returns a controldict entry.
    #     """
    #     # Return the given variable
    #     if not self.__isloaded:
    #         self._read(self.__filename)
    #
    #     if varname in self.__controldict:
    #         return self.__controldict[varname]
    #     return None
    #
    # def get_values(self, varname):
    #     """Get values for a control file variable.
    #
    #     Args:
    #         varname: The name of the control file variable.
    #
    #     Returns:
    #         Returns the value(s) associated with the control file variable
    #     """
    #     if not self.__isloaded:
    #         self._read(self.__filename)
    #
    #     thevar = self.get_var(varname)['values']
    #
    #     if thevar is not None:
    #         if len(thevar) == 1:
    #             return thevar[0]
    #         else:
    #             return thevar
    #     else:
    #         return None
    #
    # def replace_values(self, varname, newvals):
    #     if not self.__isloaded:
    #         self._read(self.__filename)
    #
    #     thevar = self.get_var(varname)
    #
    #     if isinstance(newvals, list):
    #         pass
    #     else:
    #         # Convert newvals to a list
    #         newvals = [newvals]
    #
    #     thevar['values'] = newvals
    #
    #     if varname in ['statVar_names', 'statVar_element'] and len(thevar['values']) != self.get_values('nstatVars'):
    #         # update nstatvars
    #         self.replace_values('nstatVars', len(newvals))
    #
    #         # Check if size of newvals array matches the oldvals array
    #         # if len(newvals) == len(thevar['values']):
    #         #     # Size of arrays match so replace the oldvals with the newvals
    #         #     thevar['values'] = newvals
    #         # else:
    #         #     print "ERROR: Size of oldval array and size of newval array don't match"

    # def write_control_file(self, filename):
    #     # Write the parameters out to a file
    #     if not self.__isloaded:
    #         self._read(self.__filename)
    #
    #     outfile = open(filename, 'w')
    #
    #     for hh in self.__header:
    #         # Write out any header stuff
    #         outfile.write('%s\n' % hh)
    #
    #     # Now write out the Parameter category
    #     # order = ['name', 'dimnames', 'valuetype', 'values']
    #     order = ['valuetype', 'values']
    #
    #     # Control file parameters may change. We'll use ctl_order to insure
    #     # certain parameters are always ordered, but will be followed by any
    #     # remaining non-ordered parameters.
    #     curr_ctl = set(self.__controldict.keys())
    #     curr_order = set(ctl_order)
    #     unordered_set = curr_ctl.difference(curr_order)
    #
    #     # Add parameters that are missing in the ordered set at the end of the list
    #     ctl_order.extend(list(unordered_set))
    #
    #     for kk in ctl_order:
    #         if kk in self.__controldict:
    #             vv = self.__controldict[kk]
    #         else:
    #             continue
    #
    #         valnum = len(vv['values'])
    #         valtype = vv['valuetype']
    #
    #         # Set a format string based on the valtype
    #         if valtype == 1:
    #             fmt = '%d\n'
    #         elif valtype == 2:
    #             fmt = '%f\n'
    #         else:
    #             fmt = '%s\n'
    #
    #         # Write the delimiter before the variable name
    #         outfile.write('%s\n' % VAR_DELIM)
    #         outfile.write('%s\n' % kk)
    #
    #         for item in order:
    #             # Write each variable out separated by VAR_DELIM
    #             val = vv[item]
    #
    #             if item == 'valuetype':
    #                 # valnum (which is computed) must be written before valuetype
    #                 outfile.write('%d\n' % valnum)
    #                 outfile.write('%d\n' % val)
    #             elif item == 'values':
    #                 # Write one value per line
    #                 for xx in val:
    #                     outfile.write(fmt % xx)
    #     outfile.close()
# ***** END class control()
