#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import numpy as np
from collections import OrderedDict

from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.constants import ctl_order, ctl_variable_modules, ctl_implicit_modules, \
                             DATA_TYPES, VAR_DELIM


class ControlVariable(object):
    """
    Class object for a control variable.
    """
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18
    # Description: Class object for a control variable.

    def __init__(self, name=None, datatype=None, default=None, description=None,
                 valid_values=None, value_repr=None):

        self.__name = name
        self.__datatype = datatype
        self.__default = default
        self.__description = description
        self.__valid_values = valid_values  # Possible valid values
        self.__value_repr = value_repr  # What do the valid_values represent (e.g. flag, parameter, etc)?
        self.__associated_value = None  # Based on a value what's the associated valid_value?
        self.__values = None

    def __str__(self):
        outstr = 'name: {}\ndatatype: {}\n'.format(self.name, self.datatype)

        if self.default is not None:
            outstr += 'default: {}\n'.format(self.default)

        outstr += 'Size of data: '
        if self.values is not None:
            outstr += '{}\n'.format(self.values.size)
        else:
            outstr += '<empty>\n'

        return outstr

    @property
    def associated_values(self):
        assoc_vals = []
        if self.size > 1:
            for xx in self.values:
                for vv in self.__valid_values[xx]:
                    assoc_vals.append(vv)
        else:
            for vv in self.__valid_values[str(self.values)]:
                assoc_vals.append(vv)

        return assoc_vals

    @property
    def datatype(self):
        """Returns the datatype of the control variable"""
        return self.__datatype

    @datatype.setter
    def datatype(self, dtype):
        """Sets the datatype for the control variable

        :param dtype: The datatype for the control variable (1-Integer, 2-Float, 3-Double, 4-String)
        """
        if dtype in DATA_TYPES:
            self.__datatype = dtype
        else:
            print('WARNING: Datatype, {}, is not valid.'.format(dtype))

    @property
    def default(self):
        if self.__default is not None:
            if self.__default.size > 1:
                return self.__default
            else:
                return self.__default[0]
        else:
            return None

    @default.setter
    def default(self, value):
        """Set the default value for the control variable

        :param value: The value to use
        """
        # Convert datatype first
        datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
                         3: self.__str_to_float, 4: self.__str_to_str}

        if self.__datatype in DATA_TYPES.keys():
            value = datatype_conv[self.__datatype](value)
        else:
            raise TypeError('Defined datatype {} for control variable {} is not valid'.format(self.__datatype, self.__name))

        self.__default = np.array(value)

    @property
    def name(self):
        """Returns the name of the control variable"""
        return self.__name

    @property
    def size(self):
        """Returns the number of values for the control variable"""
        if self.__values is not None:
            return self.__values.size
        elif self.__default is not None:
            return self.__default.size
        else:
            return 0


    @property
    def valid_values(self):
        return self.__valid_values

    @valid_values.setter
    def valid_values(self, data):
        if isinstance(data, dict):
            self.__valid_values = data

    @property
    def value_repr(self):
        return self.__value_repr

    @value_repr.setter
    def value_repr(self, data):
        self.__value_repr = data

    @property
    def values(self):
        if self.__values is not None:
            if self.__values.size > 1:
                return self.__values
            else:
                return self.__values[0]
        else:
            return self.default

    @values.setter
    def values(self, data):
        # Convert datatype first
        datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
                         3: self.__str_to_float, 4: self.__str_to_str}

        if self.__datatype in DATA_TYPES.keys():
            data = datatype_conv[self.__datatype](data)
        else:
            raise TypeError('Defined datatype {} for parameter {} is not valid'.format(self.__datatype,
                                                                                       self.__name))

        # Convert to ndarray
        self.__values = np.array(data)

    def __str_to_float(self, data):
        """Convert strings to a floats

        :returns: array of floats
        """

        # Convert provide list of data to float
        if isinstance(data, str):
            return [float(data)]
        else:
            try:
                return [float(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    def __str_to_int(self, data):
        """Converts strings to integers

        :returns: array of integers
        """

        if isinstance(data, str):
            return [int(data)]
        else:
            # Convert list of data to integer
            try:
                return [int(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    def __str_to_str(self, data):
        """Null op for string to string conversion

        :returns: data unchanged"""
        # nop for list of strings
        if isinstance(data, str):
            return [data]
        else:
            return data


class Control(object):
    """
    Class object for a collection of control variables.
    """
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self):
        # Container to hold dicionary of ControlVariables
        self.__control_vars = OrderedDict()
        self.__header = None

    @property
    def control_variables(self):
        return self.__control_vars

    @property
    def dynamic_parameters(self):
        """Returns a list of parameters that have the dynamic flag set"""
        dyn_params = []

        for dv in self.__control_vars.keys():
            if self.get(dv).value_repr == 'parameter':
                if self.get(dv).values > 0:
                    dyn_params.extend(self.get(dv).associated_values)
                    # dyn_params.append(self.get(dv).associated_values)
        return dyn_params

    @property
    def has_dynamic_parameters(self):
        if len(self.dynamic_parameters) > 0:
            return True
        return False

    @property
    def header(self):
        """Returns any header information defined for a control object"""
        return self.__header

    @header.setter
    def header(self, info):
        if isinstance(info, list):
            self.__header = info
        else:
            self.__header = [info]

    @property
    def modules(self):
        """Returns a dictionary of modules defined in the control file"""
        mod_dict = {}

        for xx in ctl_variable_modules:
            if self.exists(xx):
                if xx == 'precip_module':
                    if self.get(xx).values == 'climate_hru':
                        mod_dict[xx] = 'precipitation_hru'
                elif xx == 'temp_module':
                    if self.get(xx).values == 'climate_hru':
                        mod_dict[xx] = 'temperature_hru'
                else:
                    mod_dict[xx] = self.get(xx).values

        # Add the modules that are implicitly included
        for xx in ctl_implicit_modules:
            if xx not in mod_dict:
                mod_dict[xx] = ctl_implicit_modules[xx]

        return mod_dict

    def add(self, name):
        """Add a control variable by name"""
        if self.exists(name):
            raise ControlError("Control variable already exists")
        self.__control_vars[name] = ControlVariable(name=name)

    def exists(self, name):
        """Checks if a given control variable exists

        :param name: The name of the control variable
        :returns: Boolean (True if control variable exists otherwise False)
        """
        return name in self.__control_vars.keys()

    def get(self, name):
        """Returns the given control variable object

        :param name: The name of the control variable
        :returns: control variable object
        """
        if self.exists(name):
            return self.__control_vars[name]
        raise ValueError('Control variable, {}, does not exist.'.format(name))

    def remove(self, name):
        """Delete a control variable if it exists

        :param name: The name of the control variable to remove
        """
        if self.exists(name):
            del self.__control_vars[name]

    def write(self, filename):
        """Write a control file"""

        outfile = open(filename, 'w')

        for hh in self.__header:
            outfile.write('{}\n'.format(hh))

        order = ['datatype', 'values']

        # Get set of variables in ctl_order that are missing from control_vars
        setdiff = set(self.__control_vars.keys()).difference(set(ctl_order))

        # Add missing control variables (setdiff) in ctl_order to the end of the list
        ctl_order.extend(list(setdiff))

        for kk in ctl_order:
            if self.exists(kk):
                cvar = self.get(kk)
            else:
                continue

            # print(kk)
            outfile.write('{}\n'.format(VAR_DELIM))
            outfile.write('{}\n'.format(kk))

            for item in order:
                if item == 'datatype':
                    outfile.write('{}\n'.format(cvar.size))
                    outfile.write('{}\n'.format(cvar.datatype))
                if item == 'values':
                    if cvar.size == 1:
                        outfile.write('{}\n'.format(cvar.values))
                    else:
                        for cval in cvar.values:
                            outfile.write('{}\n'.format(cval))

        outfile.close()

    def _read(self):
        assert False, 'Control._read() must be defined by child class'


# ***** END class control()
