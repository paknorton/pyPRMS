
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems    # , iterkeys

from collections import OrderedDict

from pyPRMS.prms_helpers import read_xml
from pyPRMS.Exceptions_custom import ConcatError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import REGIONS, NHM_DATATYPES
from pyPRMS.constants import PARAMETERS_XML


class ParamDbRegion(ParameterSet):

    """ParameterSet sub-class which works with the ParamDb stored by CONUS regions."""

    def __init__(self, paramdb_dir, verbose=False, verify=True):
        """Initialize NhmParamDb object.

        :param str paramdb_dir: path the NHMparamDb directory
        """

        super(ParamDbRegion, self).__init__(verbose=verbose, verify=verify)
        self.__paramdb_dir = paramdb_dir

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
    def available_parameters(self):
        """Get a list of parameter names in the ParameterSet.

        :returns: list of parameter names
        :rtype: list[str]
        """

        return list(self.parameters.keys())

    @property
    def segment_nhm_to_region(self):
        """Get the dictionary which maps nhm segment ids to regional segment ids.

        :returns: dictionary NHM to regional segment ids
        :rtype: dict
        """

        return self.__nhm_to_reg_seg

    @property
    def hru_nhm_to_local(self):
        """Get the dictionary which maps NHM HRU ids to local HRU ids.

        :returns: dictionary of NHM to regional HRU ids
        :rtype: dict
        """

        return self.__nhm_to_reg_hru

    @property
    def hru_nhm_to_region(self):
        """Get the dictionary which maps NHM HRU ids to their respective region.

        :returns: dictionary of NHM HRU ids to region
        :rtype: dict
        """

        return self.__nhm_reg_range_hru

    @property
    def warnings(self):
        """Get the warnings that have occurred.

        :returns: list of warnings
        :rtype: list[str]
        """

        return self.__warnings

    def _build_global_dimensions(self):
        """Populate the global dimensions object with total dimension sizes from the parameters.
        """

        for kk, pp in iteritems(self.parameters):
            for dd in pp.dimensions.values():
                if self.dimensions.exists(dd.name):
                    if self.dimensions.get(dd.name).size != dd.size:
                        print('WARNING: {}, {}={}; current dimension size={}'.format(kk, dd.name, dd.size,
                                                                                     self.dimensions.get(dd.name).size))
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
            cdir = '{}/{}/{}'.format(self.__paramdb_dir, name, rr)

            it = self._data_it('{}/{}.csv'.format(cdir, name))
            next(it)  # Skip the header row

            # Read the parameter values
            for rec in it:
                idx, val = [int(xx) for xx in rec.split(',')]

                tmp_data.append(val)
                self.__nhm_to_reg_hru[val] = idx

            self.__nhm_reg_range_hru[rr] = [min(tmp_data), max(tmp_data)]

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

        # Read in the parameters.xml file
        params_root = read_xml(global_params_file)

        # Populate parameterSet with all available parameter names
        for param in params_root.findall('parameter'):
            xml_param_name = param.get('name')

            if self.parameters.exists(xml_param_name):
                # Sometimes the global parameter file has duplicates of parameters
                self.__warnings.append('WARNING: {} is duplicated in {}'.format(xml_param_name, PARAMETERS_XML))
                continue
            else:
                self.parameters.add(xml_param_name)
                self.parameters.get(xml_param_name).datatype = NHM_DATATYPES[param.get('type')]
                self.parameters.get(xml_param_name).units = param.get('units')
                self.parameters.get(xml_param_name).model = param.get('model')
                self.parameters.get(xml_param_name).description = param.get('desc')
                self.parameters.get(xml_param_name).help = param.get('help')

                # The original paramDb by CONUS regions does not include all the
                # information for the parameters; fill it in with the
                # master_parameters object.
                if self.master_parameters is not None:
                    try:
                        master_info = self.master_parameters.parameters[xml_param_name]

                        self.parameters.get(xml_param_name).modules = master_info.modules
                        self.parameters.get(xml_param_name).default = master_info.default
                        self.parameters.get(xml_param_name).minimum = master_info.minimum
                        self.parameters.get(xml_param_name).maximum = master_info.maximum
                    except KeyError:
                        # parameter doesn't exist in master parameter list - silently fail
                        pass

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

                # -------------------------------
                # Get a total dimension size to verify the param data read in below
                xml_root = read_xml('{}/{}.xml'.format(cdir, xml_param_name))
                size = 1

                for cdim in xml_root.findall('./dimensions/dimension'):
                    size *= int(cdim.get('size'))
                # -------------------------------

                # Read the parameter values
                for rec in it:
                    idx, val = rec.split(',')

                    if xml_param_name == 'poi_gage_segment':
                        try:
                            tmp_data.append(self.__reg_to_nhm_seg[rr][int(val)])
                        except KeyError:
                            self.__warnings.append('WARNING: poi_gage_segment for local segment {} in {}  is zero'.format(idx, rr))
                            tmp_data.append(0)
                    elif xml_param_name == 'hru_deplcrv':
                        tmp_data.append(int(val) + crv_offset)
                    else:
                        tmp_data.append(val)
                if len(tmp_data) != size:
                    print('ERROR: {} ({}) mismatch between dimensions and data ({} != {})'.format(xml_param_name, rr, size, len(tmp_data)))

                try:
                    self.parameters.get(xml_param_name).concat(tmp_data)
                except ConcatError as e:
                    self.__warnings.append(e)

                crv_offset = self.parameters.get(xml_param_name).data.size

        # self.parameters['tosegment'].data = self.parameters['tosegment_nhm'].data
        # self.parameters['hru_segment'].data = self.parameters['hru_segment_nhm'].data
