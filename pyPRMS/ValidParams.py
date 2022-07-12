import io
import pkgutil
import xml.etree.ElementTree as xmlET
from typing import Optional, Sequence, Set, Union

from pyPRMS.Parameters import Parameters
from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.constants import NHM_DATATYPES


class ValidParams(Parameters):

    """Object containing master list of parameters."""

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04

    def __init__(self, filename: Optional[str] = None):
        """Create ValidParams object.

        Read an XML file of parameters to use as a master of valid PRMS
        parameters. If no filename is specified an internal library XML file
        is read.

        :param filename: Name of XML parameter file
        """

        super(ValidParams, self).__init__()

        self.filename = filename

    @property
    def filename(self) -> Optional[Union[str, None]]:
        """Get XML filename.

        Returned filename is None if reading from the library-internal XML file.

        :returns: Name of XML file
        """

        return self.__filename

    @filename.setter
    def filename(self, filename: Optional[str] = None):
        """Set the XML file name.

        If no filename is specified an library-internal XML file is read.

        :param filename: name of XML parameter file
        """

        self.__filename = filename

        if self.filename is not None:
            self.__xml_tree = xmlET.parse(self.filename)
        else:
            # Use the package parameters.xml by default
            res = pkgutil.get_data('pyPRMS', 'xml/parameters.xml')

            assert res is not None
            xml_fh = io.StringIO(res.decode('utf-8'))
            self.__xml_tree = xmlET.parse(xml_fh)

        self.__isloaded = False
        self._read()
        self.__isloaded = True

    def get_params_for_modules(self, modules: Sequence[str]) -> Set[str]:
        """Get list of unique parameters required for a given list of modules.

        :param modules: List of PRMS modules

        :returns: Set of parameter names
        """

        params_by_module = []

        for xx in self.parameters.values():
            if xx.modules is not None:
                for mm in xx.modules:
                    if mm in modules:
                        params_by_module.append(xx.name)
        return set(params_by_module)

    def _read(self):
        """Read an XML parameter file.

        The resulting Parameters object will have parameters that have no data.
        """

        xml_root = self.__xml_tree.getroot()

        # Iterate over child nodes of root
        for elem in xml_root.findall('parameter'):
            # print(elem.attrib.get('name'))
            name = elem.attrib.get('name')
            dtype = elem.find('type').text
            # print(name)
            try:
                self.add(name)

                self.get(name).datatype = NHM_DATATYPES[dtype]
                self.get(name).description = elem.find('desc').text
                self.get(name).units = elem.find('units').text

                try:
                    self.get(name).minimum = elem.find('minimum').text
                except AttributeError:
                    pass

                try:
                    self.get(name).maximum = elem.find('maximum').text
                except AttributeError:
                    pass

                try:
                    self.get(name).default = elem.find('default').text
                except AttributeError:
                    # Parameter has no default value
                    pass

                # Add dimensions for current parameter
                for cdim in elem.findall('./dimensions/dimension'):
                    try:
                        self.get(name).dimensions.add(cdim.attrib.get('name'), size=int(cdim.find('default').text))
                    except AttributeError:
                        # Dimension has no default value
                        self.get(name).dimensions.add(cdim.attrib.get('name'))

                mods = []
                for cmod in elem.findall('./modules/module'):
                    mods.append(cmod.text)

                self.get(name).modules = mods
            except ParameterError:
                # Parameter exists add any new attribute information
                pass
