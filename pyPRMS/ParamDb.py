
from __future__ import (absolute_import, division, print_function)
# # from future.utils import iteritems    # , iterkeys

# from collections import OrderedDict

from pyPRMS.prms_helpers import read_xml
# from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import NHM_DATATYPES
from pyPRMS.constants import PARAMETERS_XML, DIMENSIONS_XML


class ParamDb(ParameterSet):
    def __init__(self, paramdb_dir, verbose=False, verify=True):
        """Initialize ParamDb object.
        This object handles the monolithic parameter database.

        :param str paramdb_dir: path the ParamDb directory
        """

        super(ParamDb, self).__init__(verbose=verbose, verify=verify)
        self.__paramdb_dir = paramdb_dir
        self.__verbose = verbose

        # Read the parameters from the parameter database
        self._read()

    @property
    def available_parameters(self):
        """Get a list of parameter names in the ParameterSet.

        :returns: list of parameter names
        :rtype: list[str]
        """

        return list(self.parameters.keys())

    @staticmethod
    def _data_it(filename):
        """Get iterator to a parameter db file.

        :returns: iterator
        """

        # Read the data
        fhdl = open(filename)
        rawdata = fhdl.read().splitlines()
        fhdl.close()
        return iter(rawdata)

    def _read(self):
        """Read a paramDb file.
        """

        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types
        global_params_file = '{}/{}'.format(self.__paramdb_dir, PARAMETERS_XML)
        global_dimens_file = '{}/{}'.format(self.__paramdb_dir, DIMENSIONS_XML)

        # Read in the parameters.xml and dimensions.xml file
        params_root = read_xml(global_params_file)
        dimens_root = read_xml(global_dimens_file)

        # Populate the global dimensions from the xml file
        for xml_dim in dimens_root.findall('dimension'):
            self.dimensions.add(name=xml_dim.attrib.get('name'), size=int(xml_dim.find('size').text))

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.attrib.get('name')

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                print('WARNING: {} is duplicated in {}'.format(xml_param_name, PARAMETERS_XML))
                continue
            else:
                self.parameters.add(xml_param_name)
                self.parameters.get(xml_param_name).datatype = NHM_DATATYPES[param.find('type').text]
                self.parameters.get(xml_param_name).units = getattr(param.find('units'), 'text', None)
                self.parameters.get(xml_param_name).description = getattr(param.find('desc'), 'text', None)
                self.parameters.get(xml_param_name).help = getattr(param.find('help'), 'text', None)
                # self.parameters.get(xml_param_name).model = param.get('model')

                self.parameters.get(xml_param_name).default = getattr(param.find('default'), 'text', None)

                self.parameters.get(xml_param_name).minimum = getattr(param.find('minimum'), 'text', None)
                self.parameters.get(xml_param_name).maximum = getattr(param.find('maximum'), 'text', None)
                self.parameters.get(xml_param_name).modules = [cmod.text for cmod in param.findall('./modules/module')]

            # Read the parameter data
            tmp_data = []

            # Read parameter information
            try:
                it = self._data_it('{}/{}.csv'.format(self.__paramdb_dir, xml_param_name))
                next(it)  # Skip the header row
            except IOError:
                print('Skipping parameter: {}. File does not exist.'.format(xml_param_name))
                continue

            # Add dimensions for current parameter
            for cdim in param.findall('./dimensions/dimension'):
                dim_name = cdim.attrib.get('name')
                self.parameters.get(xml_param_name).dimensions.add(name=dim_name,
                                                                   size=self.dimensions.get(dim_name).size)

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')
                tmp_data.append(val)

            self.parameters.get(xml_param_name).data = tmp_data

            if not self.parameters.get(xml_param_name).has_correct_size():
                err_txt = 'ERROR: {} mismatch between dimensions and size of data. Removed from parameter set.'
                print(err_txt.format(xml_param_name))
                self.parameters.remove(xml_param_name)
