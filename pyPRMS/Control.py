#!/usr/bin/env python

# from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

import numpy as np
from collections import OrderedDict

from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.constants import ctl_order, ctl_variable_modules, ctl_implicit_modules, \
                             DATA_TYPES, VAR_DELIM


class ControlVariable(object):

    """
    Class object for a single control variable.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, name=None, datatype=None, default=None, description=None,
                 valid_values=None, value_repr=None):
        """Initialize a control variable object.
        """

        self.__name = name
        self.__datatype = datatype
        self.__default = default
        self.__description = description
        self.__force_default = False
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
            outstr += '{}\n'.format(self.size)
        else:
            outstr += '<empty>\n'

        return outstr

    @property
    def associated_values(self):
        """Get list of control variable names which are associated with this
        control variable.

        :returns: associated control variables
        :rtype: list[str]
        """

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
        """Get the datatype of the control variable.

        :returns: datatype
        :rtype: int
        """

        return self.__datatype

    @datatype.setter
    def datatype(self, dtype):
        """Sets the datatype of the control variable.

        :param int dtype: The datatype for the control variable (1-Integer, 2-Float, 3-Double, 4-String)
        """

        if dtype in DATA_TYPES:
            self.__datatype = dtype
        else:
            print('WARNING: Datatype, {}, is not valid.'.format(dtype))

    @property
    def force_default(self):
        """Get logical value which indicates whether the default value for a
        control variable should always be used instead of the current value.

        :rtype: bool
        """

        return self.__force_default

    @force_default.setter
    def force_default(self, value):
        """Set (or unset) forced use of default value.

        :param bool value: new force_default value
        """

        self.__force_default = bool(value)

    @property
    def default(self):
        """Get default value for control variable.

        :returns: current default value
        :rtype: int or float or str
        """

        if self.__default is not None:
            if self.__default.size > 1:
                return self.__default
            else:
                return self.__default[0]
        else:
            return None

    @default.setter
    def default(self, value):
        """Set the default value for the control variable.

        :param value: new default value
        :type value: int or float or str
        """

        # Convert datatype first
        datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
                         3: self.__str_to_float, 4: self.__str_to_str}

        if self.__datatype in DATA_TYPES.keys():
            value = datatype_conv[self.__datatype](value)
        else:
            err_txt = 'Defined datatype {} for control variable {} is not valid'
            raise TypeError(err_txt.format(self.__datatype, self.__name))

        self.__default = np.array(value)

    @property
    def name(self):
        """Returns the name of the control variable.

        :returns: name of control variable
        :rtype: str
        """

        return self.__name

    @property
    def size(self):
        """Get the number of values for the control variable.

        :returns: number of values
        :rtype: int
        """

        if self.__values is not None:
            return self.__values.size
        elif self.__default is not None:
            return self.__default.size
        else:
            return 0

    @property
    def valid_values(self):
        """Get the values that are valid for the control variable.

        :returns: valid values for the control variable
        :rtype: dict
        """

        return self.__valid_values

    @valid_values.setter
    def valid_values(self, data):
        """Set the values that are valid for the control variable.

        :param dict data: valid values for the control variable
        """

        if isinstance(data, dict):
            self.__valid_values = data

    @property
    def value_repr(self):
        """Get what the control variable value represents.

        A control variable value can represent a flag, interval, or parameter.

        :returns: control variable representation value
        :rtype: str
        """

        return self.__value_repr

    @value_repr.setter
    def value_repr(self, data):
        """Set the control variable representation.

        :param data: representation value
        :type data: str or None
        """

        self.__value_repr = data

    @property
    def values(self):
        """Get the values for the control variable.

        If force_default is True then the default value is returned regardless
        of what values is set to; otherwise, current values is returned.

        :returns: value(s) of control variable
        :rtype: list[int or str] or int or float or str
        """

        if self.__values is not None:
            if self.__force_default:
                return self.default
            elif self.__values.size > 1:
                return self.__values
            else:
                return self.__values[0]
        else:
            return self.default

    @values.setter
    def values(self, data):
        """Set the values for the control variable.

        :param data: new value(s)
        :type data: list[str] or str
        """

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

    @staticmethod
    def __str_to_float(data):
        """Convert strings to a floats.

        :param data: value(s)
        :type data: list[str] or str

        :returns: array of floats
        :rtype: list[float]
        """

        # Convert provide list of data to float
        if isinstance(data, str):
            return [float(data)]
        else:
            try:
                return [float(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    @staticmethod
    def __str_to_int(data):
        """Converts strings to integers.

        :param data: value(s)
        :type data: list[str] or str

        :returns: array of integers
        :rtype: list[int]
        """

        if isinstance(data, str):
            return [int(data)]
        else:
            # Convert list of data to integer
            try:
                return [int(vv) for vv in data]
            except ValueError as ve:
                print(ve)

    @staticmethod
    def __str_to_str(data):
        """Null op for string-to-string conversion.

        :param data: value(s)
        :type data: list[str] or str

        :returns: unmodified array of data
        :rtype: list[str]
        """

        # nop for list of strings
        if isinstance(data, str):
            data = [data]

        # 2019-05-22 PAN: For python 3 force string type to byte
        #                 otherwise they are treated as unicode
        return data
        # return [dd.encode() for dd in data]


class Control(object):

    """
    Class object for a collection of control variables.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self):
        """Create Control object.
        """

        # Container to hold dicionary of ControlVariables
        self.__control_vars = OrderedDict()
        self.__header = None

    @property
    def control_variables(self):
        """Get control variable objects.

        :returns: control variable objects
        :rtype: collections.OrderedDict[str, ControlVariable]
        """

        return self.__control_vars

    @property
    def dynamic_parameters(self):
        """Get parameter names that have the dynamic flag set.

        :returns: list of parameter names
        :rtype: list[str]
        """

        dyn_params = []

        for dv in self.__control_vars.keys():
            if self.get(dv).value_repr == 'parameter':
                if self.get(dv).values > 0:
                    dyn_params.extend(self.get(dv).associated_values)
                    # dyn_params.append(self.get(dv).associated_values)
        return dyn_params

    @property
    def has_dynamic_parameters(self):
        """Indicates if any dynamic parameters have been requested.

        :returns: True if dynamic parameters are required
        :rtype: bool
        """

        if len(self.dynamic_parameters) > 0:
            return True
        return False

    @property
    def header(self):
        """Get header information defined for a control object.

        This is typically taken from the first two lines of a control file.

        :returns: header information
        :rtype: bool
        """

        return self.__header

    @header.setter
    def header(self, info):
        """Set the header information.

        :param info: header line(s)
        :type info: list[str] or str
        """

        if isinstance(info, list):
            self.__header = info
        else:
            self.__header = [info]

    @property
    def modules(self):
        """Get the modules defined in the control file.

        :returns: defined modules
        :rtype: dict[str, str]
        """

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
        """Add a control variable by name.

        :param str name: name of the control variable

        :raises ControlError: if control variable already exists
        """

        if self.exists(name):
            raise ControlError("Control variable already exists")
        self.__control_vars[name] = ControlVariable(name=name)

    def exists(self, name):
        """Checks if a given control variable exists.

        :param str name: name of the control variable
        :returns: True if control variable exists otherwise False
        :rtype: bool
        """

        return name in self.__control_vars.keys()

    def get(self, name):
        """Returns the given control variable object.

        :param str name: name of the control variable

        :returns: control variable object
        :rtype: ControlVariable

        :raises ValueError: if control variable does not exist
        """

        if self.exists(name):
            return self.__control_vars[name]
        raise ValueError('Control variable, {}, does not exist.'.format(name))

    def remove(self, name):
        """Delete a control variable if it exists.

        :param str name: name of the control variable
        """

        if self.exists(name):
            del self.__control_vars[name]

    def write(self, filename):
        """Write a control file.

        :param str filename: name of control file to create
        """

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
                        # print(type(cvar.values))
                        if isinstance(cvar.values, np.bytes_):
                            print("BYTES")
                            outfile.write('{}\n'.format(cvar.values.decode()))
                        else:
                            outfile.write('{}\n'.format(cvar.values))
                    else:
                        for cval in cvar.values:
                            outfile.write('{}\n'.format(cval))

        outfile.close()

    def _read(self):
        """Abstract function for reading.
        """
        assert False, 'Control._read() must be defined by child class'


# ***** END class control()
