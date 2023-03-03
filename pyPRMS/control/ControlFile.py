#!/usr/bin/env python3

# import io
# import pkgutil
# import xml.etree.ElementTree as xmlET
# from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType

from typing import Optional, Union
from ..constants import DATA_TYPES, VAR_DELIM
from .Control import Control
from ..Exceptions_custom import ControlError


class ControlFile(Control):
    """
    Class which handles the processing of PRMS control files.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18
    # Description: Class object to handle reading and writing PRMS
    #              control files.

    def __init__(self, filename: str, verbose: Optional[bool] = False,
                 version:Optional[Union[str, int]] = 5):
        super(ControlFile, self).__init__(verbose=verbose, version=version)

        self.__verbose = verbose
        self.__isloaded = False
        self.filename = filename


    @property
    def filename(self) -> str:
        """Get control filename.

        :returns: Name of control file
        """
        return self.__filename

    @filename.setter
    def filename(self, filename: str):
        """Set the name of the control file and read it in.

        :param filename: Name of the control file
        """

        self.__isloaded = False
        self.__filename = filename
        self._read()

    def _read(self):
        """Read contents of control file.

        Reads the contents of a control file into the class.
        """

        if self.__verbose:
            chk_vars = []

        self.__isloaded = False

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

                if self.__verbose:
                    if varname in chk_vars:
                        print(f'WARNING: {varname} already exists')
                    chk_vars.append(varname)

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
                        print(f'varname: {varname} value type and defined type {DATA_TYPES[valuetype]} don\'t match')

                if len(vals) != numval:
                    print(f'ERROR: Not enough values provided for {varname}')
                    print(f'       Expected {numval}, got {len(vals)}')

                try:
                    cnt = numval
                    while next(it) != VAR_DELIM:
                        cnt += 1

                    if cnt > numval:
                        print(f'WARNING: Too many values specified for {varname}')
                        print(f'      {numval} expected, {cnt} given')
                        print(f'       Keeping first {numval} values')
                except StopIteration:
                    # Hit the end of the file
                    pass

                try:
                    self.add(name=varname, datatype=(valuetype))
                except ControlError:
                    pass

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
