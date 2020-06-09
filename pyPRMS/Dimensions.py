
from collections import OrderedDict
from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType
import xml.etree.ElementTree as xmlET

from pyPRMS.Dimension import Dimension
from pyPRMS.prms_helpers import read_xml


class Dimensions(object):

    """Container of Dimension objects."""

    def __init__(self, verbose=False):
        """Create ordered dictionary to contain Dimension objects."""
        self.__dimensions = OrderedDict()
        self.__verbose = verbose

    def __str__(self):
        outstr = ''
        if len(self.__dimensions) == 0:
            outstr = '<empty>'
        else:
            for kk, vv in self.__dimensions.items():
                outstr += f'{kk}: {vv}\n'
        return outstr

    def __getattr__(self, name):
        # print('ATTR: {}'.format(name))
        return getattr(self.__dimensions, name)

    def __getitem__(self, item):
        """Get named dimension."""
        return self.__dimensions[item]

    @property
    def dimensions(self) -> OrderedDictType[str, Dimension]:
        """Get ordered dictionary of Dimension objects.

        :returns: OrderedDict of Dimension objects
        :rtype: collections.OrderedDict[str, Dimension]
        """

        return self.__dimensions

    @property
    def ndims(self) -> int:
        """Get number of dimensions.

        :returns: number of dimensions
        :rtype: int
        """

        return len(self.__dimensions)

    @property
    def xml(self) -> xmlET.Element:
        """Get xml element for the dimensions.

        :returns: XML element for the dimensions
        :rtype: xmlET.Element
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

    def add(self, name: str, size: int = 0):
        """Add a new dimension.

        :param str name: name of the dimension
        :param int size: size of the dimension
        """

        # This method adds a dimension if it doesn't exist
        # Duplicate dimension names are silently ignored
        # TODO: check for valid dimension size for ndays, nmonths, and one
        if name not in self.__dimensions:
            try:
                self.__dimensions[name] = Dimension(name=name, size=size)
            except ValueError as err:
                if self.__verbose:
                    print(err)
                else:
                    pass
        # else:
        #     # TODO: Should this raise an error?
        #     print('Dimension {} already exists...skipping add name'.format(name))

    def add_from_xml(self, filename: str):
        """Add one or more dimensions from an xml file.

        :param str filename: name of xml file to read
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
            name = cdim.get('name')
            size = int(cdim.get('size'))

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

        :param str name: name of the dimension
        :returns: True if dimension exists, otherwise False
        :rtype: bool
        """

        return name in self.dimensions.keys()

    def get(self, name: str) -> Dimension:
        """Get dimension.

        :param str name: name of the dimension

        :returns: dimension
        :rtype: Dimension

        :raises ValueError: if dimension does not exist
        """

        if self.exists(name):
            return self.__dimensions[name]
        raise ValueError(f'Dimension, {name}, does not exist.')

    def remove(self, name: str):
        """Remove dimension.

        :param str name: dimension name
        """

        if self.exists(name):
            del self.__dimensions[name]

    def tostructure(self) -> Dict[str, Dict[str, int]]:
        """Get data structure of Dimensions data for serialization.

        :returns: dictionary of dimension names and sizes
        :rtype: dict
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
        :rtype: xmlET.Element
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
        """Add a new dimension.

        :param str name: name of the dimension
        :param int size: size of the dimension
        """

        # Restrict number of dimensions for parameters
        if self.ndims == 2:
            raise ValueError('Number of dimensions greater than 2 is not supported')
        super().add(name, size)

    def add_from_xml(self, filename: str):
        """Add one or more dimensions from an xml file.

        Add or grow dimensions from XML information. This version also checks dimension position.

        :param str filename: name of the xml file

        :raises ValueError: if existing dimension position is altered
        """

        # Add dimensions and grow dimension sizes from xml information for a parameter
        # This information is found in xml files for each region for each parameter
        # No attempt is made to verify whether each region for a given parameter
        # has the same or same number of dimensions.
        xml_root = read_xml(filename)

        for cdim in xml_root.findall('./dimensions/dimension'):
            name = cdim.get('name')
            size = int(cdim.get('size'))
            pos = int(cdim.get('position')) - 1

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

        :param int index: The 0-based position of the dimension.
        :returns: Size of the dimension.
        :rtype: int
        :raises ValueError: if index is greater than number dimensions for the parameter
        """

        if index < len(self.dimensions.items()):
            try:
                # Python 2.7.x
                return self.dimensions.items()[index][1].size
            except TypeError:
                # Python 3.x
                return list(self.dimensions.items())[index][1].size
        raise ValueError(f'Parameter has no dimension at index {index}')

    def get_position(self, name: str) -> int:
        """Get 0-based index position of a dimension.

        :param str name: name of the dimension

        :returns: index position of dimension
        :rtype: int
        """

        # TODO: method name should be index() ??
        return list(self.dimensions.keys()).index(name)

    def tostructure(self) -> Dict[str, Dict[str, int]]:
        """Get dictionary structure of the dimensions.

        :returns: dictionary of Dimensions names, sizes, and positions
        :rtype: dict
        """

        ldims = super(ParamDimensions, self).tostructure()
        for kk, vv in ldims.items():
            vv['position'] = self.get_position(kk)
        return ldims
