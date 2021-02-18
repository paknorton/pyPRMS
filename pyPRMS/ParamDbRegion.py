
from collections import OrderedDict
from pathlib import Path
# from typing import Any,  Union, Dict, Iterator, List, OrderedDict as OrderedDictType, Set
from typing import Dict, Iterator, List, Set

import numpy as np
import pandas as pd

from pyPRMS.prms_helpers import read_xml
# from pyPRMS.Exceptions_custom import ConcatError, ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import REGIONS, NHM_DATATYPES
from pyPRMS.constants import DATATYPE_TO_DTYPE, PARAMETERS_XML


class ParamDbRegion(ParameterSet):

    """ParameterSet sub-class which works with the ParamDb stored by CONUS regions."""

    def __init__(self, paramdb_dir: str, verbose=False, verify=True):
        """Initialize NhmParamDb object.

        :param str paramdb_dir: path the NHMparamDb directory
        """

        super(ParamDbRegion, self).__init__(verbose=verbose, verify=verify)
        self.__paramdb_dir = paramdb_dir

        self.__verbose = verbose
        self.__warnings = []
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

        if len(self.__warnings) > 0:
            print('There were {} warnings while reading'.format(len(self.__warnings)))

    @property
    def segment_nhm_to_region(self) -> Dict[int, int]:
        """Get the dictionary which maps nhm segment ids to regional segment ids.

        :returns: dictionary NHM to regional segment ids
        :rtype: dict
        """

        return self.__nhm_to_reg_seg

    @property
    def hru_nhm_to_local(self) -> Dict[int, int]:
        """Get the dictionary which maps NHM HRU ids to local HRU ids.

        :returns: dictionary of NHM to regional HRU ids
        :rtype: dict
        """

        return self.__nhm_to_reg_hru

    @property
    def hru_nhm_to_region(self) -> Dict[int, int]:
        """Get the dictionary which maps NHM HRU ids to their respective region.

        :returns: dictionary of NHM HRU ids to region
        :rtype: dict
        """

        return self.__nhm_reg_range_hru

    @property
    def warnings(self) -> List[str]:
        """Get the warnings that occurred when the parameter database was read.

        :returns: list of warnings
        :rtype: list[str]
        """

        return self.__warnings

    def _build_global_dimensions(self):
        """Populate the global dimensions object with total dimension sizes from the parameters.
        """

        for kk, pp in self.parameters.items():
            for dd in pp.dimensions.values():
                if self.dimensions.exists(dd.name):
                    if self.dimensions.get(dd.name).size != dd.size:
                        print(f'WARNING: {kk}, {dd.name}={dd.size}; ' +
                              f'current dimension size={self.dimensions.get(dd.name).size}')
                else:
                    self.dimensions.add(name=dd.name, size=dd.size)

    def _create_seg_maps(self):
        """Create mapping dictionaries NHM-to-regional segments IDs (and vice-versa).
        """
        name = 'nhm_seg'
        self.__reg_to_nhm_seg = {}
        self.__nhm_to_reg_seg = {}

        for rr in REGIONS:
            # Read parameter information
            cdir = f'{self.__paramdb_dir}/{name}/{rr}'

            it = self._data_it(f'{cdir}/{name}.csv')
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
        """Create the mapping dictionaries for HRUs.

        Creates two dictionaries which map: 1) NHM HRU ids to regional HRU ids,
        and 2) the range of NHM HRU ids for each region.
        """

        name = 'nhm_id'

        self.__nhm_reg_range_hru = {}
        self.__nhm_to_reg_hru = OrderedDict()
        # self.__nhm_to_reg_hru = {}

        for rr in REGIONS:
            tmp_data = []

            # Read parameter information
            cdir = f'{self.__paramdb_dir}/{name}/{rr}'

            it = self._data_it(f'{cdir}/{name}.csv')
            next(it)  # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = [int(xx) for xx in rec.split(',')]

                tmp_data.append(val)
                self.__nhm_to_reg_hru[val] = idx

            self.__nhm_reg_range_hru[rr] = [min(tmp_data), max(tmp_data)]

    @staticmethod
    def _data_it(filename: str) -> Iterator:
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
        global_params_file = f'{self.__paramdb_dir}/{PARAMETERS_XML}'

        # Read in the parameters.xml file
        params_root = read_xml(global_params_file)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.get('name')

            if xml_param_name in ['hru_deplcrv', 'poi_gage_segment']:
                # Skip processing
                continue

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                self.__warnings.append(f'WARNING: {xml_param_name} is duplicated in {PARAMETERS_XML}')
                continue

            # Add the parameter
            if self.master_parameters is not None:
                # Prefer the master set of parameters for information
                try:
                    self.parameters.add(xml_param_name, info=self.master_parameters[xml_param_name])
                except ValueError:
                    print(f'WARNING: {xml_param_name} is not a valid PRMS parameter; skipping.')
                    continue
            else:
                self.parameters.add(xml_param_name, datatype=NHM_DATATYPES[param.get('type')],
                                    units=param.get('units'), model=param.get('model'),
                                    description=param.get('desc'), help=param.get('help'))

            self._set_parameter_dimension_size(xml_param_name)
            curr_param = self.parameters.get(xml_param_name)

            all_param_files = sorted(Path(f'{self.__paramdb_dir}/{xml_param_name}').rglob('*.csv'))

            df_list = [pd.read_csv(f, usecols=[1], squeeze=True,
                                   dtype={1: DATATYPE_TO_DTYPE[self.parameters.get(xml_param_name).datatype]})
                       for f in all_param_files]

            if curr_param.ndims == 2:
                # Handle 2D parameters a bit differently; each region has to be reshaped before
                # being concatenated together.
                param_data = [dd.values.reshape((-1, curr_param.dimensions.get_dimsize_by_index(1),), order='F')
                              for dd in df_list]
                param_data = np.vstack(param_data)
            else:
                param_data = pd.concat(df_list)
            curr_param.data = param_data

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Deal with the special case: poi_gage_segment
        tmp_data = []
        xml_param_name = 'poi_gage_segment'

        print(f'Read special: {xml_param_name}')

        # Add the parameter
        self.parameters.add(xml_param_name, info=self.master_parameters[xml_param_name])

        self._set_parameter_dimension_size(xml_param_name)

        for rr in REGIONS:
            # Read parameter information
            cdir = f'{self.__paramdb_dir}/{xml_param_name}/{rr}'

            it = self._data_it('{}/{}.csv'.format(cdir, xml_param_name))
            next(it)    # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')

                try:
                    tmp_data.append(self.__reg_to_nhm_seg[rr][int(val)])
                except KeyError:
                    self.__warnings.append(f'WARNING: poi_gage_segment for local segment {idx} in {rr} is zero')
                    tmp_data.append(0)

        if len(tmp_data) != self.parameters.get(xml_param_name).size:
            print(f'ERROR: {xml_param_name} mismatch between dimensions and ' +
                  f'data ({self.parameters.get(xml_param_name).size} != {len(tmp_data)})')
        else:
            self.parameters.get(xml_param_name).data = tmp_data

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Deal with the special case: hru_deplcrv
        tmp_data = []
        xml_param_name = 'hru_deplcrv'

        print(f'Read special: {xml_param_name}')

        # Add the parameter
        self.parameters.add(xml_param_name, info=self.master_parameters[xml_param_name])

        crv_offset = 0

        self._set_parameter_dimension_size(xml_param_name)

        for rr in REGIONS:
            # Read parameter information
            cdir = f'{self.__paramdb_dir}/{xml_param_name}/{rr}'

            it = self._data_it('{}/{}.csv'.format(cdir, xml_param_name))
            next(it)    # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = rec.split(',')
                tmp_data.append(int(val) + crv_offset)

            crv_offset = len(tmp_data)

        if len(tmp_data) != self.parameters.get(xml_param_name).size:
            print(f'ERROR: {xml_param_name} mismatch between dimensions and ' +
                  f'data ({self.parameters.get(xml_param_name).size} != {len(tmp_data)})')
        else:
            self.parameters.get(xml_param_name).data = tmp_data

        # self.parameters['tosegment'].data = self.parameters['tosegment_nhm'].data
        # self.parameters['hru_segment'].data = self.parameters['hru_segment_nhm'].data

    def _set_parameter_dimension_size(self, name: str):
        # Set dimensions size for entire NHM for parameter
        for rr in REGIONS:
            # Read parameter information
            cdir = f'{self.__paramdb_dir}/{name}/{rr}'

            # Add/grow dimensions for current parameter
            self.parameters.get(name).dimensions.add_from_xml(f'{cdir}/{name}.xml')

    def _get_parameter_dimension_names(self, name: str) -> Set[str]:
        """Returns a set of dimension names for a parameter"""
        dtmp = []
        for rr in REGIONS:
            cdir = f'{self.__paramdb_dir}/{name}/{rr}'
            xml_root = read_xml(f'{cdir}/r01/{name}.xml')

            for cdim in xml_root.findall('./dimensions/dimension'):
                dtmp.append(cdim.get('name'))

        return set(dtmp)
