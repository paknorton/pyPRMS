#!/usr/bin/env python3

import io
import operator
import pkgutil
import xml.etree.ElementTree as xmlET

import numpy as np
from collections import OrderedDict

try:
    from typing import Dict, List, Optional, OrderedDict as OrderedDictType, Sequence, Union
except ImportError:
    # pre-python 3.7.2
    from typing import Union, Dict, List, MutableMapping as OrderedDictType, Sequence   # type: ignore

from pyPRMS.prms_helpers import version_info
from pyPRMS.ControlVariable import ControlVariable
from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.constants import ctl_order, ctl_variable_modules, ctl_implicit_modules, \
                             ctl_summary_modules, VAR_DELIM

cond_check = {'=': operator.eq,
              '>': operator.gt,
              '<': operator.lt}

class Control(object):
    """
    Class object for a collection of control variables.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, verbose: Optional[bool] = False, version: Optional[Union[str, int]] = 5):
        """Create Control object.
        """

        # Container to hold dicionary of ControlVariables
        self.__control_vars = OrderedDict()
        self.__header = None
        self.__verbose = verbose

        # First read control.xml from the library
        # This makes sure any missing variables from the control file
        # end up with default values
        xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/control.xml').decode('utf-8'))
        xml_tree = xmlET.parse(xml_fh)
        xml_root = xml_tree.getroot()

        for elem in xml_root.findall('control_param'):
            var_version = version_info(elem.attrib.get('version'))
            depr_version = version_info(elem.attrib.get('deprecated'))
            name = elem.attrib.get('name')

            if (var_version.major is not None and var_version.major > version):
                if self.__verbose:
                    print(f'{name} rejected by version')
                continue
            if (depr_version.major is not None and depr_version.major <= version):
                if self.__verbose:
                    print(f'{name} rejected by deprecation version')
                continue

            datatype = int(elem.find('type').text)

            self.add(name=name, datatype=datatype)

            if name in ['start_time', 'end_time']:
                # Hack to handle PRMS approach to dates
                dt = elem.find('default').text.split('-')
                if len(dt) < 6:
                    # pad short date with zeros for hms
                    dt.extend(['0' for _ in range(6 - len(dt))])
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

    def __getitem__(self, item: str) -> ControlVariable:
        """Get ControlVariable object for a variable.

        :param item: name of control file variable
        :returns: ControlVariable object
        """
        return self.get(item)

    @property
    def control_variables(self) -> OrderedDictType[str, ControlVariable]:
        """Get control variable objects.

        :returns: control variable objects
        """
        return self.__control_vars

    @property
    def dynamic_parameters(self) -> List[str]:
        """Get list parameter names for which a dynamic flag set.

        :returns: list of parameter names
        """

        dyn_params = []

        for dv in self.__control_vars.keys():
            cvar = self.get(dv)

            if cvar.value_repr == 'parameter' and (isinstance(cvar.values, int) or isinstance(cvar.values, np.int64)):
                # dynamic parameter flags should always be integers
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

        for xx in ctl_variable_modules:
            if self.exists(xx):
                if xx == 'precip_module':
                    if self.get(xx).values == 'climate_hru':
                        mod_dict[xx] = 'precipitation_hru'
                    else:
                        mod_dict[xx] = str(self.get(xx).values)
                elif xx == 'temp_module':
                    if self.get(xx).values == 'climate_hru':
                        mod_dict[xx] = 'temperature_hru'
                    else:
                        mod_dict[xx] = str(self.get(xx).values)
                else:
                    mod_dict[xx] = str(self.get(xx).values)

        # Add the modules that are implicitly included
        for xx in ctl_implicit_modules:
            if xx not in mod_dict:
                mod_dict[xx] = ctl_implicit_modules[xx]

        return mod_dict

    @property
    def summary_modules(self) -> List[str]:
        """Get list of summary modules in PRMS
        """

        summary_modules = {'basin_sum': '',
                           'basin_summary': 'basinOutON_OFF > 0',
                           'map_results': 'mapOutON_OFF > 0',
                           'nhru_summary': 'nhruOutON_OFF > 0',
                           'nsegment_summary': 'nsegmentOutON_OFF > 0',
                           'nsub_summary': 'nsubOutON_OFF > 0',
                           'subbasin': 'subbasin_flag = 1'}

        active_modules = []

        for cmod, cond in summary_modules.items():
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

    def add(self, name: str, datatype: int):
        """Add a control variable by name.

        :param name: Name of the control variable
        :param datatype: The datatype of the control variable

        :raises ControlError: if control variable already exists
        """

        if self.exists(name):
            raise ControlError("Control variable already exists")
        self.__control_vars[name] = ControlVariable(name=name, datatype=datatype)

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
                    if item == 'datatype':
                        outfile.write(f'{cvar.size}\n')
                        outfile.write(f'{cvar.datatype}\n')
                    if item == 'values':
                        if cvar.size == 1:
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
