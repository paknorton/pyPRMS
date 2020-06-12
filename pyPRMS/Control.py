#!/usr/bin/env python3

import numpy as np
from collections import OrderedDict
from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType, Sequence

from pyPRMS.ControlVariable import ControlVariable
from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.constants import ctl_order, ctl_variable_modules, ctl_implicit_modules, \
                             DATA_TYPES, VAR_DELIM


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
    def control_variables(self) -> OrderedDictType[str, ControlVariable]:
        """Get control variable objects.

        :returns: control variable objects
        :rtype: collections.OrderedDict[str, ControlVariable]
        """

        return self.__control_vars

    @property
    def dynamic_parameters(self) -> List[str]:
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
    def has_dynamic_parameters(self) -> bool:
        """Indicates if any dynamic parameters have been requested.

        :returns: True if dynamic parameters are required
        :rtype: bool
        """
        return len(self.dynamic_parameters) > 0

    @property
    def header(self) -> Union[List[str], None]:
        """Get header information defined for a control object.

        This is typically taken from the first two lines of a control file.

        :returns: header information
        """

        return self.__header

    @header.setter
    def header(self, info: Union[Sequence[str], str]):
        """Set the header information.

        :param info: header line(s)
        :type info: list[str] or str
        """

        if isinstance(info, list):
            self.__header = info
        else:
            self.__header = [info]

    @property
    def modules(self) -> Dict[str, str]:
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

    def add(self, name: str):
        """Add a control variable by name.

        :param str name: name of the control variable

        :raises ControlError: if control variable already exists
        """

        if self.exists(name):
            raise ControlError("Control variable already exists")
        self.__control_vars[name] = ControlVariable(name=name)

    def exists(self, name: str) -> bool:
        """Checks if a given control variable exists.

        :param str name: name of the control variable
        :returns: True if control variable exists otherwise False
        :rtype: bool
        """

        return name in self.__control_vars.keys()

    def get(self, name: str) -> ControlVariable:
        """Returns the given control variable object.

        :param str name: name of the control variable

        :returns: control variable object
        :rtype: ControlVariable

        :raises ValueError: if control variable does not exist
        """

        if self.exists(name):
            return self.__control_vars[name]
        raise ValueError(f'Control variable, {name}, does not exist.')

    def remove(self, name: str):
        """Delete a control variable if it exists.

        :param str name: name of the control variable
        """

        if self.exists(name):
            del self.__control_vars[name]

    def write(self, filename: str):
        """Write a control file.

        :param str filename: name of control file to create
        """

        outfile = open(filename, 'w')

        for hh in self.__header:
            outfile.write(f'{hh}\n')

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
            outfile.write(f'{VAR_DELIM}\n')
            outfile.write(f'{kk}\n')

            for item in order:
                if item == 'datatype':
                    outfile.write(f'{cvar.size}\n')
                    outfile.write(f'{cvar.datatype}\n')
                if item == 'values':
                    if cvar.size == 1:
                        # print(type(cvar.values))
                        if isinstance(cvar.values, np.bytes_):
                            print("BYTES")
                            outfile.write(f'{cvar.values.decode()}\n')
                        else:
                            outfile.write(f'{cvar.values}\n')
                    else:
                        for cval in cvar.values:
                            outfile.write(f'{cval}\n')

        outfile.close()

    def _read(self):
        """Abstract function for reading.
        """
        assert False, 'Control._read() must be defined by child class'


# ***** END class control()
