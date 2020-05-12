
import numpy as np
import os
import pandas as pd
from pyPRMS.prms_helpers import read_xml
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

    def _read_old(self):
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
            self.dimensions.add(name=xml_dim.attrib.get('name'), size=xml_dim.find('size').text)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.attrib.get('name')

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                print('WARNING: {} is duplicated in {}'.format(xml_param_name, PARAMETERS_XML))
                continue
            else:
                self.parameters.add(name=xml_param_name,
                                    datatype=NHM_DATATYPES[param.find('type').text],
                                    units=getattr(param.find('units'), 'text', None),
                                    description=getattr(param.find('desc'), 'text', None),
                                    help=getattr(param.find('help'), 'text', None),
                                    default=getattr(param.find('default'), 'text', None),
                                    minimum=getattr(param.find('minimum'), 'text', None),
                                    maximum=getattr(param.find('maximum'), 'text', None),
                                    modules=[cmod.text for cmod in param.findall('./modules/module')])
                # # self.parameters.get(xml_param_name).model = param.get('model')

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

    def _read(self):
        """Read a paramDb file.
        """

        datatype_conv = {1: np.int, 2: np.float, 3: np.float, 4: np.str_}

        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types
        global_params_file = '{}/{}'.format(self.__paramdb_dir, PARAMETERS_XML)
        global_dimens_file = '{}/{}'.format(self.__paramdb_dir, DIMENSIONS_XML)

        # Read in the parameters.xml and dimensions.xml file
        params_root = read_xml(global_params_file)
        dimens_root = read_xml(global_dimens_file)

        # Populate the global dimensions from the xml file
        for xml_dim in dimens_root.findall('dimension'):
            self.dimensions.add(name=xml_dim.attrib.get('name'), size=xml_dim.find('size').text)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.attrib.get('name')
            curr_file = f'{self.__paramdb_dir}/{xml_param_name}.csv'

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                print('WARNING: {} is duplicated in {}'.format(xml_param_name, PARAMETERS_XML))
                continue

            if os.path.exists(curr_file):
                self.parameters.add(name=xml_param_name,
                                    datatype=NHM_DATATYPES[param.find('type').text],
                                    units=getattr(param.find('units'), 'text', None),
                                    description=getattr(param.find('desc'), 'text', None),
                                    help=getattr(param.find('help'), 'text', None),
                                    default=getattr(param.find('default'), 'text', None),
                                    minimum=getattr(param.find('minimum'), 'text', None),
                                    maximum=getattr(param.find('maximum'), 'text', None),
                                    modules=[cmod.text for cmod in param.findall('./modules/module')])
                # # self.parameters.get(xml_param_name).model = param.get('model')

                # Add dimensions from the global dimensions for current parameter
                for cdim in param.findall('./dimensions/dimension'):
                    dim_name = cdim.attrib.get('name')
                    self.parameters.get(xml_param_name).dimensions.add(name=dim_name,
                                                                       size=self.dimensions.get(dim_name).size)

                tmp_data = pd.read_csv(curr_file, skiprows=0, usecols=[1],
                                       dtype={1: datatype_conv[self.parameters.get(xml_param_name).datatype]},
                                       squeeze=True).to_numpy()
                # tmp_data = np.loadtxt(curr_file, delimiter=',',
                #                       skiprows=1, usecols=[1], ndmin=1,
                #                       dtype=datatype_conv[self.parameters.get(xml_param_name).datatype])

                self.parameters.get(xml_param_name).data = tmp_data

                if not self.parameters.get(xml_param_name).has_correct_size():
                    err_txt = 'ERROR: {} mismatch between dimensions and size of data. Removed from parameter set.'
                    print(err_txt.format(xml_param_name))
                    self.parameters.remove(xml_param_name)
            else:
                print('Skipping parameter: {}. File does not exist.'.format(xml_param_name))