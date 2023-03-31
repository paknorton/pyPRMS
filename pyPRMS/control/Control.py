#!/usr/bin/env python3

import io
import numpy as np
import operator
import pkgutil
import re
import xml.etree.ElementTree as xmlET

from collections import OrderedDict

from typing import Dict, List, Optional, OrderedDict as OrderedDictType, Sequence, Union

from ..prms_helpers import version_info
from .ControlVariable import ControlVariable
from ..Exceptions_custom import ControlError
from ..constants import ctl_order, ctl_variable_modules, ctl_implicit_modules, VAR_DELIM, PTYPE_TO_PRMS_TYPE

cond_check = {'=': operator.eq,
              '>': operator.gt,
              '<': operator.lt}

class Control(object):
    """
    Class object for a collection of control variables.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, metadata, verbose: Optional[bool] = False, version: Optional[Union[str, int]] = 5):
        """Create Control object.
        """

        # Container to hold dicionary of ControlVariables
        self.__control_vars = OrderedDict()
        self.__header = None
        self.__verbose = verbose

        for cvar, cvals in metadata.items():
            # print(cvar)
            self.add(name=cvar, meta=cvals)

        print('Pre-populate done')

    def __getitem__(self, item: str) -> ControlVariable:
        """Get ControlVariable object for a variable.

        :param item: name of control file variable
        :returns: ControlVariable object
        """
        return self.get(item)

    @property
    def control_data(self):
        """Dictionary of data for each control variable"""
        data_dict = {}

        for kk, vv in self.__control_vars.items():
            data_dict[kk] = vv.values

        return data_dict

    @property
    def control_variables(self) -> OrderedDictType[str, ControlVariable]:
        """Get control variable objects.

        :returns: control variable objects
        """
        return self.__control_vars

    @property
    def dynamic_parameters(self) -> List[str]:
        """Get list of parameter names for which a dynamic flag set.

        :returns: list of parameter names
        """

        dyn_params = []

        for dv in self.__control_vars.keys():
            cvar = self.get(dv)

            if cvar.meta['valid_value_type'] == 'parameter' and (isinstance(cvar.values, np.int32) or isinstance(cvar.values, np.int64)):
                # Dynamic parameter flags should always be integers
                if cvar.values > 0:
                    dyn_params.extend(cvar.associated_values)
                    # dyn_params.append(self.get(dv).associated_values)
        return dyn_params

    @property
    def has_dynamic_parameters(self) -> bool:
        """Indicates if any dynamic parameters have been requested.

        :returns: True if dynamic parameters are required
        """
        return len(self.dynamic_parameters) > 0

    @property
    def header(self) -> Union[List[str], None]:
        """Get header information defined for a control object.

        This is typically taken from the first two lines of a control file.

        :returns: Header information from control file
        """
        return self.__header

    @header.setter
    def header(self, info: Union[Sequence[str], str]):
        """Set the header information.

        :param info: list or string of header line(s)
        """

        if isinstance(info, list):
            self.__header = info
        else:
            self.__header = [info]

    @property
    def modules(self) -> Dict[str, str]:
        """Get the modules defined in the control file.

        :returns: dictionary of control variable, module name pairs
        """

        mod_dict = {}

        for vv in self.control_variables.values():
            if vv.meta.get('valid_value_type', '') == 'module':
                mname = vv.values

                if vv.name == 'precip_module':
                    if vv.values == 'climate_hru':
                        mname = 'precipitation_hru'
                if vv.name == 'temp_module':
                    if vv.values == 'climate_hru':
                        mname = 'temperature_hru'

                mod_dict[vv.name] = mname

        # Add the modules that are implicitly included
        for mtype, mname in ctl_implicit_modules.items():
            if mtype not in mod_dict:
                mod_dict[mtype] = mname

        return mod_dict

    @property
    def additional_modules(self) -> List[str]:
        """Get list of summary modules in PRMS
        """

        # TODO: module_requirements should be added to metadata?
        module_requirements = {'basin_sum': 'print_debug = 4',
                               'basin_summary': 'basinOutON_OFF > 0',
                               'map_results': 'mapOutON_OFF > 0',
                               'nhru_summary': 'nhruOutON_OFF > 0',
                               'nsegment_summary': 'nsegmentOutON_OFF > 0',
                               'nsub_summary': 'nsubOutON_OFF > 0',
                               'stream_temp': 'stream_temp_flag > 0',
                               'subbasin': 'subbasin_flag = 1'}

        active_modules = []

        for cmod, cond in module_requirements.items():
            if self._check_condition(cond):
                active_modules.append(cmod)

        return active_modules

    def _check_condition(self, cstr: str) -> bool:
        """Takes a string of the form '<control_var> <op> <value>' and checks
        if the condition is True
        """
        if len(cstr) == 0:
            return True

        var, op, value = cstr.split(' ')
        value = int(value)

        if self.exists(var):
            return cond_check[op](self.get(var).values, value)
        return False

    def add(self, name: str, meta=None):
        """Add a control variable by name.

        :param name: Name of the control variable
        :param datatype: The datatype of the control variable

        :raises ControlError: if control variable already exists
        """

        if self.exists(name):
            raise ControlError("Control variable already exists")
        self.__control_vars[name] = ControlVariable(name=name, meta=meta)
        # self.__control_vars[name] = ControlVariable(name=name, datatype=datatype, meta=meta)

    def exists(self, name: str) -> bool:
        """Checks if control variable exists.

        :param name: Name of the control variable
        :returns: True if control variable exists otherwise False
        """
        return name in self.__control_vars.keys()

    def get(self, name: str) -> ControlVariable:
        """Returns the given control variable object.

        :param name: Name of the control variable
        :returns: Control variable object

        :raises ValueError: if control variable does not exist
        """

        if self.exists(name):
            return self.__control_vars[name]
        raise ValueError(f'Control variable, {name}, does not exist.')

    def remove(self, name: str):
        """Delete a control variable if it exists.

        :param name: Name of the control variable
        """

        if self.exists(name):
            del self.__control_vars[name]

    def write(self, filename: str):
        """Write a control file.

        :param filename: Name of control file to create
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

                outfile.write(f'{VAR_DELIM}\n')
                outfile.write(f'{kk}\n')

                for item in order:
                    if cvar.meta['datatype'] == 'datetime':
                        date_tmp = [int(xx) for xx in re.split(r'[-T:.]+', str(cvar.values))[0:6]]

                        if item == 'datatype':
                            outfile.write(f'{len(date_tmp)}\n')
                            outfile.write(f'{PTYPE_TO_PRMS_TYPE[cvar.meta["datatype"]]}\n')
                        if item == 'values':
                            for cval in date_tmp:
                                outfile.write(f'{cval}\n')
                    else:
                        if item == 'datatype':
                            outfile.write(f'{cvar.values.size}\n')
                            outfile.write(f'{PTYPE_TO_PRMS_TYPE[cvar.meta["datatype"]]}\n')
                        if item == 'values':
                            if cvar.meta['context'] == 'scalar':
                                # Single-values (e.g. int, float, str)
                                # print(type(cvar.values))
                                if isinstance(cvar.values, np.bytes_):
                                    print("BYTES")
                                    outfile.write(f'{cvar.values.decode()}\n')
                                else:
                                    outfile.write(f'{cvar.values}\n')
                            else:
                                # Multiple-values
                                if isinstance(cvar.values, np.ndarray):
                                    for cval in cvar.values:
                                        outfile.write(f'{cval}\n')


        outfile.close()

    def _read(self):
        """Abstract function for reading.
        """
        assert False, 'Control._read() must be defined by child class'


# ***** END class control()
