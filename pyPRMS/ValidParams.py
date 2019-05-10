
from __future__ import (absolute_import, division, print_function)

import io
import pkgutil
import xml.etree.ElementTree as xmlET
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.constants import NHM_DATATYPES


class ValidParams(ParameterSet):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-01-06
    # Description: Object for the database of valid input parameters

    def __init__(self, filename=None):
        super(ValidParams, self).__init__()

        self.__filename = filename

        if filename:
            self.__xml_tree = xmlET.parse(self.__filename)
        else:
            # Use the package file, parameters.xml, by default
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

    def get_params_for_modules(self, modules):
        """Returns a list of unique parameters required for a given list of modules"""

        params_by_module = []

        for xx in self.parameters.values():
            for mm in xx.modules:
                if mm in modules:
                    params_by_module.append(xx.name)
        return set(params_by_module)

    def _read(self):
        """Read a parameter.xml file to create a parameter set with no data"""

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
