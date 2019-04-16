
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

import glob
import os
import xml.dom.minidom as minidom
import io
import pkgutil
import xml.etree.ElementTree as xmlET
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.constants import NHM_DATATYPES
from pyPRMS.constants import PARNAME_DATATYPES


class ValidParams_v2(ParameterSet):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-01-06
    # Description: Object for the database of valid input parameters

    def __init__(self, filename=None):
        super(ValidParams_v2, self).__init__()

        # Mapping of certain parameters to their correct module
        # self.__mod_map = {'hru_pansta': ['potet_pan'],
        #                   'tmin_adj': ['temp_1sta', 'temp_laps', 'ide_dist', 'xyz_dist'],
        #                   'tmax_adj': ['temp_1sta', 'temp_laps', 'ide_dist', 'xyz_dist'],
        #                   'hru_tsta': ['temp_1sta', 'temp_laps'],
        #                   'basin_tsta': ['temp_1sta', 'temp_laps', 'temp_dist2'],
        #                   'psta_elev': ['precip_laps', 'ide_dist', 'xyz_dist'],
        #                   'tsta_elev': ['temp_1sta', 'temp_laps', 'temp_dist2', 'ide_dist', 'xyz_dist'],
        #                   'elevlake_init': ['muskingum_lake'],
        #                   'gw_seep_coef': ['muskingum_lake'],
        #                   'lake_evap_adj': ['muskingum_lake'],
        #                   'lake_hru': ['muskingum_lake'],
        #                   'lake_hru_id': ['muskingum_lake'],
        #                   'lake_seep_elev': ['muskingum_lake'],
        #                   'lake_type': ['muskingum_lake']}

        self.__filename = filename

        if filename:
            self.__xml_tree = xmlET.parse(self.__filename)
        else:
            # Use the package parameters.xml by default
            xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/parameters.xml').decode('utf-8'))
            self.__xml_tree = xmlET.parse(xml_fh)

        # TODO: need more robust logic here; currently no way to handle failures
        self.__isloaded = False
        self._read()
        self.__isloaded = True

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, filename):
        self.__filename = filename

        if filename:
            self.__xml_tree = xmlET.parse(self.__filename)
        else:
            # Use the package parameters.xml by default
            xml_fh = io.StringIO(pkgutil.get_data('pyPRMS', 'xml/parameters.xml').decode('utf-8'))
            self.__xml_tree = xmlET.parse(xml_fh)

        self.__isloaded = False
        self._read()
        self.__isloaded = True

    def _read(self):
        """Read a parametetr.xml file to create a parameter set with no data"""

        xml_root = self.__xml_tree.getroot()

        # Iterate over child nodes of root
        for elem in xml_root.findall('parameter'):
            # print(elem.attrib.get('name'))
            name = elem.attrib.get('name')
            dtype = elem.find('type').text
            # print(name)
            try:
                self.parameters.add(name)

                self.parameters.get(name).datatype = NHM_DATATYPES[dtype]
                self.parameters.get(name).description = elem.find('desc').text
                self.parameters.get(name).maximum = elem.find('maximum').text

                # Add dimensions for current parameter
                for cdim in elem.findall('./dimensions/dimension'):
                    self.parameters.get(name).dimensions.add(cdim.attrib.get('name'))

                for cmod in elem.findall('./modules/module'):
                    self.parameters.get(name).modules = cmod.text
            except ParameterError:
                # Parameter exists add any new attribute information
                pass
