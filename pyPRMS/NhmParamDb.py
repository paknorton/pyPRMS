
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems    # , iterkeys


# import numpy as np

from pyPRMS.prms_helpers import read_xml
# from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import REGIONS, NHM_DATATYPES
from pyPRMS.constants import PARAMETERS_XML


class NhmParamDb(ParameterSet):
    def __init__(self, paramdb_dir):
        super(NhmParamDb, self).__init__()
        self.__paramdb_dir = paramdb_dir

        # Build mappings between national and regional ids
        self.__reg_to_nhm_seg = {}
        self.__nhm_to_reg_seg = {}
        self._create_seg_maps()

        self.__nhm_to_reg_hru = {}
        self.__nhm_reg_range_hru = {}
        self._create_hru_maps()

        # Read the parameters from the parameter database
        self._read()

        # Populate the global dimensions information
        self._build_global_dimensions()

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
                self.parameters.get(xml_param_name).units = param.get('units')
                self.parameters.get(xml_param_name).model = param.get('model')
                self.parameters.get(xml_param_name).description = param.get('desc')
                self.parameters.get(xml_param_name).help = param.get('help')

            # Get dimensions information for each of the parameters
            for rr in REGIONS:
                # Read parameter information
                cdir = '{}/{}/{}'.format(self.__paramdb_dir, xml_param_name, rr)

                # Add/grow dimensions for current parameter
                self.parameters.get(xml_param_name).dimensions.add_from_xml('{}/{}.xml'.format(cdir, xml_param_name))

            crv_offset = 0  # Only used for hru_deplcrv

            for rr in REGIONS:
                # Read the parameter data
                tmp_data = []

                # Read parameter information
                cdir = '{}/{}/{}'.format(self.__paramdb_dir, xml_param_name, rr)

                it = self._data_it('{}/{}.csv'.format(cdir, xml_param_name))
                next(it)    # Skip the header row

                # Read the parameter values
                for rec in it:
                    idx, val = rec.split(',')

                    if xml_param_name == 'poi_gage_segment':
                        try:
                            tmp_data.append(self.__reg_to_nhm_seg[rr][int(val)])
                        except KeyError:
                            print('WARNING: poi_gage_segment for local segment {} in {}  is zero'.format(idx, rr))
                            tmp_data.append(0)
                    elif xml_param_name == 'hru_deplcrv':
                        tmp_data.append(int(val) + crv_offset)
                    else:
                        tmp_data.append(val)

                self.parameters.get(xml_param_name).concat(tmp_data)
                crv_offset = self.parameters.get(xml_param_name).data.size
