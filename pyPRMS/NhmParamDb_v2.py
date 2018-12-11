
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems    # , iterkeys


# import numpy as np

from pyPRMS.prms_helpers import read_xml
# from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import REGIONS, NHM_DATATYPES
from pyPRMS.constants import PARAMETERS_XML, DIMENSIONS_XML


class NhmParamDb_v2(ParameterSet):
    def __init__(self, paramdb_dir):
        super(NhmParamDb_v2, self).__init__()
        self.__paramdb_dir = paramdb_dir

        # Build mappings between national and regional ids
        # self.__reg_to_nhm_seg = {}
        # self.__nhm_to_reg_seg = {}
        # self._create_seg_maps()

        self.__nhm_to_reg_hru = {}
        self.__nhm_reg_range_hru = {}
        # self._create_hru_maps()

        # Read the parameters from the parameter database
        self._read()

        # Populate the global dimensions information
        # self._build_global_dimensions()

    @property
    def available_parameters(self):
        return self.parameters.keys()

    @property
    def segment_nhm_to_region(self):
        return self.__nhm_to_reg_seg

    @property
    def hru_nhm_to_local(self):
        return self.__nhm_to_reg_hru

    @property
    def hru_nhm_to_region(self):
        return self.__nhm_reg_range_hru

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

    def _create_seg_maps(self):
        name = 'nhm_seg'
        self.__reg_to_nhm_seg = {}
        self.__nhm_to_reg_seg = {}

        for rr in REGIONS:
            # Read parameter information
            cdir = '{}/{}/{}'.format(self.__paramdb_dir, name, rr)

            it = self._data_it('{}/{}.csv'.format(cdir, name))
            next(it)  # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = [int(xx) for xx in rec.split(',')]

                # Add regional segment id to nhm_seg mapping by region
                if rr not in self.__reg_to_nhm_seg:
                    self.__reg_to_nhm_seg[rr] = {}
                self.__reg_to_nhm_seg[rr][idx] = val

                # Add nhm_seg to regional segment id mapping
                self.__nhm_to_reg_seg[val] = idx

    def _create_hru_maps(self):
        name = 'nhm_id'

        self.__nhm_reg_range_hru = {}
        self.__nhm_to_reg_hru = {}

        for rr in REGIONS:
            tmp_data = []

            # Read parameter information
            cdir = '{}/{}/{}'.format(self.__paramdb_dir, name, rr)

            it = self._data_it('{}/{}.csv'.format(cdir, name))
            next(it)  # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = [int(xx) for xx in rec.split(',')]

                tmp_data.append(val)
                self.__nhm_to_reg_hru[val] = idx

            self.__nhm_reg_range_hru[rr] = [min(tmp_data), max(tmp_data)]

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

            # Read the parameter data
            tmp_data = []

            # Read parameter information
            try:
                it = self._data_it('{}/{}.csv'.format(self.__paramdb_dir, xml_param_name))
                next(it)  # Skip the header row
            except IOError:
                # print('Skipping parameter: {}. File does not exist.'.format(xml_param_name))
                continue

            # Add the parameter information
            self.parameters.add(xml_param_name)
            self.parameters.get(xml_param_name).datatype = NHM_DATATYPES[param.find('type').text]
            self.parameters.get(xml_param_name).units = param.find('units').text
            self.parameters.get(xml_param_name).description = param.find('desc').text
            self.parameters.get(xml_param_name).help = param.find('help').text

            try:
                self.parameters.get(xml_param_name).default = param.find('default').text
            except AttributeError:
                # print('\tNo default set')
                pass

            self.parameters.get(xml_param_name).minimum = param.find('minimum').text
            self.parameters.get(xml_param_name).maximum = param.find('maximum').text
            self.parameters.get(xml_param_name).modules = [cmod.text for cmod in param.findall('./modules/module')]

            # Add dimensions for current parameter
            for cdim in param.findall('./dimensions/dimension'):
                dim_name = cdim.attrib.get('name')
                self.parameters.get(xml_param_name).dimensions.add(name=dim_name, size=self.dimensions.get(dim_name).size)

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')
                tmp_data.append(val)

            self.parameters.get(xml_param_name).data = tmp_data

            if not self.parameters.get(xml_param_name).has_correct_size():
                print('ERROR: {} mismatch between dimensions and size of data. Removed from parameter set.'.format(xml_param_name))
                self.parameters.remove(xml_param_name)
