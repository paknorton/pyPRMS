from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems    # , iterkeys

# import xml.etree.ElementTree as xmlET
# import numpy as np

from pyPRMS.prms_helpers import read_xml
# from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import PARAMETERS_XML, NHM_DATATYPES    # , DIMENSIONS_XML


class ParamDb(ParameterSet):
    def __init__(self, paramdb_dir):
        super(ParamDb, self).__init__()
        self.__paramdb_dir = paramdb_dir

        # Build mappings between national and regional ids
        # self.__reg_to_nhm_seg = {}
        # self.__nhm_to_reg_seg = {}
        # self._create_seg_maps()
        #
        # self.__nhm_to_reg_hru = {}
        # self.__nhm_reg_range_hru = {}
        # self._create_hru_maps()

        # Read the parameters from the parameter database
        self._read()

        # Populate the global dimensions information
        self._build_global_dimensions()

    @property
    def available_parameters(self):
        return self.parameters.keys()

    def _build_global_dimensions(self):
        """Populate the global dimensions object with total dimension sizes from the parameters"""
        for kk, pp in iteritems(self.parameters):
            for dd in pp.dimensions.values():
                if self.dimensions.exists(dd.name):
                    if self.dimensions.get(dd.name).size != dd.size:
                        print('WARNING: {}, {}={}; current dimension size={}'.format(kk, dd.name, dd.size,
                                                                                     self.dimensions.get(dd.name).size))
                else:
                    self.dimensions.add(name=dd.name, size=dd.size)

    def _data_it(self, filename):
        """Returns iterator to parameter db file"""
        # Read the data
        fhdl = open(filename)
        rawdata = fhdl.read().splitlines()
        fhdl.close()
        return iter(rawdata)

    def _read(self):
        # Get the parameters available from the parameter database
        # Returns a dictionary of parameters and associated units and types
        global_params_file = '{}/{}'.format(self.__paramdb_dir, PARAMETERS_XML)

        # Read in the parameters.xml file
        params_root = read_xml(global_params_file)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.get('name')

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                print('WARNING: {} is duplicated in {}'.format(xml_param_name, PARAMETERS_XML))
                continue
            else:
                self.parameters.add(xml_param_name)
                self.parameters.get(xml_param_name).datatype = NHM_DATATYPES[param.get('type')]
                # self.parameters.get(xml_param_name).units = param.get('units')

            # Get dimensions information for each of the parameters
            # Read parameter information
            cdir = '{}'.format(self.__paramdb_dir)

            # Add/grow dimensions for current parameter
            self.parameters.get(xml_param_name).dimensions.add_from_xml('{}/{}.xml'.format(cdir, xml_param_name))

            # Read the parameter data
            tmp_data = []

            # Read parameter information
            it = self._data_it('{}/{}.csv'.format(cdir, xml_param_name))
            next(it)    # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')
                tmp_data.append(val)

            self.parameters.get(xml_param_name).concat(tmp_data)
