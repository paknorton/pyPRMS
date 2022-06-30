
from collections import OrderedDict

try:
    from typing import cast, Dict, Optional, OrderedDict as OrderedDictType, Union
except ImportError:
    # pre-python 3.7.2
    from typing import cast, Dict, Optional, MutableMapping as OrderedDictType   # type: ignore

import xml.etree.ElementTree as xmlET

from pyPRMS.Dimension import Dimension
from pyPRMS.prms_helpers import read_xml


class Dimensions(object):
    """Container of Dimension objects."""
    __dimensions: OrderedDictType

    def __init__(self, verbose: Optional[bool] = False,
                 verify: Optional[bool] = True):
        """Create ordered dictionary containing Dimension objects.

        :param verbose: Output additional debug information
        :param verify: Enforce valid dimension names (default=True)
        """
        self.__dimensions = OrderedDict()
        self.__verbose = verbose
        self.__verify = verify

    def __str__(self) -> str:
        """Pretty-print dimensions.

        :returns: Pretty-print string of dimensions
        """

        outstr = ''
        if len(self.__dimensions) == 0:
            outstr = '<empty>'
        else:
            for kk, vv in self.__dimensions.items():
                outstr += f'{kk}: {vv}\n'
        return outstr

    def __getattr__(self, name: str) -> str:
        """Get named dimension.

        :param name: name of dimensions
        :returns: dimension object
        """

        # print('ATTR: {}'.format(name))
        return getattr(self.__dimensions, name)

    def __getitem__(self, item: str) -> Dimension:
        """Get named dimension.

        :param item: name of dimension
        :returns: Dimension object
        """
        return self.__dimensions[item]

    @property
    def dimensions(self) -> OrderedDictType[str, Dimension]:
        """Get ordered dictionary of Dimension objects.

        :returns: OrderedDict of Dimension objects
        """
        return self.__dimensions

    @property
    def ndims(self) -> int:
        """Get number of dimensions.

        :returns: Number of dimensions
        """

        return len(self.__dimensions)

    @property
    def xml(self) -> xmlET.Element:
        """Get xml element for the dimensions.

        :returns: XML element for the dimensions
        """

        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in self.dimensions.items():
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            xmlET.SubElement(dim_sub, 'size').text = str(vv.size)
            # dim_sub.set('size', str(vv.size))
        return dims_xml

    def add(self, name: str, size: int):
        """Add a new Dimension object.

        :param name: Name of the dimension
        :param size: Size of the dimension
        """

        # This method adds a dimension if it doesn't exist
        # Duplicate dimension names are silently ignored
        # TODO: check for valid dimension size for ndays, nmonths, and one
        if name not in self.__dimensions:
            try:
                self.__dimensions[name] = Dimension(name=name, size=size)
            except ValueError as err:
                if self.__verify:
                    print(err)
                else:
                    pass
        # else:
        #     # TODO: Should this raise an error?
        #     print('Dimension {} already exists...skipping add name'.format(name))

    def add_from_xml(self, filename: str):
        """Add one or more dimensions from an xml file.

        :param filename: Name of xml file to read
        """

        # Add dimensions and grow dimension sizes from xml information for a parameter
        # This information is found in xml files for each region for each parameter
        # No attempt is made to verify whether each region for a given parameter
        # has the same or same number of dimensions.
        xml_root = read_xml(filename)

        # TODO: We can't guarantee the order of the dimensions in the xml file
        #       so we should make sure dimensions are added in the correct order
        #       dictated by the position attribute.
        #       1) read all dimensions in the correct 'position'-dictated order into a list
        #       2) add dimensions in list to the dimensions ordereddict
        for cdim in xml_root.findall('./dimensions/dimension'):
            name = cast(str, cdim.get('name'))
            size = cast(int, cdim.get('size'))
            # name = cdim.get('name')
            # size = int(cdim.get('size'))

            if name not in self.__dimensions:
                try:
                    self.__dimensions[name] = Dimension(name=name, size=size)
                except ValueError as err:
                    print(err)
            else:
                if name not in ['nmonths', 'ndays', 'one']:
                    # NOTE: This will always try to grow a dimension if it already exists!
                    self.__dimensions[name].size += size

    def exists(self, name: str) -> bool:
        """Check if dimension exists.

        :param name: Name of the dimension
        :returns: True if dimension exists, otherwise False
        """

        return name in self.dimensions.keys()

    def get(self, name: str) -> Dimension:
        """Get dimension.

        :param name: Name of the dimension

        :returns: Dimension object

        :raises ValueError: if dimension does not exist
        """

        if self.exists(name):
            return self.__dimensions[name]
        raise ValueError(f'Dimension, {name}, does not exist.')

    def remove(self, name: str):
        """Remove Dimension object.

        :param name: name of dimension to remove
        """

        if self.exists(name):
            del self.__dimensions[name]

    def tostructure(self) -> Dict[str, Dict[str, int]]:
        """Get data structure of Dimensions data for serialization.

        :returns: dictionary of dimension names and sizes
        """

        dims = {}
        for kk, vv in self.dimensions.items():
            dims[kk] = {'size': vv.size}
        return dims


class ParamDimensions(Dimensions):
    """Container for parameter dimensions.

    This object adds tracking of dimension position.
    """

    @property
    def xml(self) -> xmlET.Element:
        """Get xml for the dimensions.

        :returns: XML element of the dimensions
        """

        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in self.dimensions.items():
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            xmlET.SubElement(dim_sub, 'position').text = str(self.get_position(kk)+1)
            xmlET.SubElement(dim_sub, 'size').text = str(vv.size)

            # dim_sub.set('position', str(self.get_position(kk)+1))
            # dim_sub.set('size', str(vv.size))
        return dims_xml

    def add(self, name: str, size: int = 0):
        """Add a new Dimension object.

        :param name: Name of the dimension
        :param size: Size of the dimension
        """

        # Restrict number of dimensions for parameters
        super().add(name, size)

        if self.ndims > 2:
            raise ValueError('Number of dimensions greater than 2 is not supported')

    def add_from_xml(self, filename: str):
        """Add one or more dimensions from an xml file.

        Add or grow dimensions from XML information. This version also checks dimension position.

        :param filename: Name of the xml file

        :raises ValueError: if existing dimension position is altered
        """

        # Add dimensions and grow dimension sizes from xml information for a parameter
        # This information is found in xml files for each region for each parameter
        # No attempt is made to verify whether each region for a given parameter
        # has the same or same number of dimensions.
        xml_root = read_xml(filename)

        for cdim in xml_root.findall('./dimensions/dimension'):
            name = cast(str, cdim.get('name'))
            size = cast(int, cdim.get('size'))
            pos = cast(int, cdim.get('position')) - 1
            # name = cdim.get('name')
            # size = int(cdim.get('size'))
            # pos = int(cdim.get('position')) - 1

            if name not in self.dimensions:
                try:
                    self.dimensions[name] = Dimension(name=name, size=size)
                except ValueError as err:
                    print(err)
            else:
                curr_pos = list(self.dimensions.keys()).index(name)

                if curr_pos != pos:
                    # This indicates a problem in one of the paramdb files
                    raise ValueError(f'{name}: Attempted position change from {curr_pos} to {pos}')
                else:
                    if name not in ['nmonths', 'ndays', 'one']:
                        # NOTE: This will always try to grow a dimension if it already exists!
                        self.dimensions[name].size += size

    # noinspection PyUnresolvedReferences
    def get_dimsize_by_index(self, index: int) -> int:
        """Return size of dimension at the given index.

        :param index: The 0-based position of the dimension
        :returns: Size of the dimension

        :raises ValueError: if index is greater than number dimensions for the parameter
        """

        if index < len(self.dimensions.items()):
            return list(self.dimensions.items())[index][1].size
        raise ValueError(f'Parameter has no dimension at index {index}')

    def get_position(self, name: str) -> int:
        """Get 0-based index position of a dimension.

        :param name: name of the dimension

        :returns: Zero-based Index position of dimension
        """

        # TODO: method name should be index() ??
        return list(self.dimensions.keys()).index(name)

    def tostructure(self) -> Dict[str, Dict[str, int]]:
        """Get dictionary structure of the dimensions.

        :returns: dictionary of Dimensions names, sizes, and positions
        """

        ldims = super(ParamDimensions, self).tostructure()
        for kk, vv in ldims.items():
            vv['position'] = self.get_position(kk)
        return ldims
