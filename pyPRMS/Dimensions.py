
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

from collections import OrderedDict
import xml.etree.ElementTree as xmlET

from pyPRMS.constants import DIMENSION_NAMES
from pyPRMS.prms_helpers import read_xml


def _valid_dimension_name(name):
    """Check if given dimension name is valid for PRMS.

    :param str name: dimension name
    :returns: True if dimension name is valid otherwise False
    :rtype: bool
    """

    return name in DIMENSION_NAMES


class Dimension(object):

    """Defines a single dimension."""

    def __init__(self, name=None, size=0, description=None):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param str name: The name of the dimension
        :param int size: The size of the dimension
        :param description: Description of the dimension
        :type description: str or None
        """

        self.__name = None
        self.__size = None
        self.__description = None
        self.name = name
        self.size = size
        self.description = description

    @property
    def name(self):
        """Name of the dimension.

        :returns: Name of the dimension
        :rtype: str
        """

        return self.__name

    @name.setter
    def name(self, name):
        """Sets the name of the dimension.

        :param str name: Name of the dimension
        :raises ValueError: if dimension name is not a valid PRMS dimension
        """

        if _valid_dimension_name(name):
            self.__name = name
        else:
            # TODO: Should this raise an error?
            raise ValueError('Dimension name, {}, is not a valid PRMS dimension name'.format(name))

    @property
    def size(self):
        """Size of the dimension.

        :returns: size of the dimension
        :rtype: int
        """

        return self.__size

    @size.setter
    def size(self, value):
        """Set the size of the dimension.

        :param int value: size of the dimension
        :raises ValueError: if dimension size in not a positive integer
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError('Dimension size must be a positive integer')
        self.__size = value

    @property
    def description(self):
        """Description for the dimension.

        :returns: description for the dimension
        :rtype: str
        """

        return self.__description

    @description.setter
    def description(self, descstr):
        """Set the description of the dimension.

        :param str descstr: description string
        """

        self.__description = descstr

    def __repr__(self):
        return 'Dimension(name={}, size={!r})'.format(self.name, self.size)

    def __iadd__(self, other):
        """Add a number to dimension size.

        :param int other: integer value

        :returns: dimension size
        :rtype: int

        :raises ValueError: if type of parameter is not an integer
        """

        # augment in-place addition so the instance plus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        self.__size += other
        return self

    def __isub__(self, other):
        """Subtracts integer from dimension size.

        :param int other: integer value

        :returns: dimension size
        :rtype: int

        :raises ValueError: if type of parameter is not an integer
        :raises ValeuError: if parameter is not a positive integer
        """

        # augment in-place addition so the instance minus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        if self.__size - other < 0:
            raise ValueError('Dimension size must be positive')
        self.__size -= other
        return self


class Dimensions(object):

    """Container of Dimension objects."""

    def __init__(self):
        """Create ordered dictionary to contain Dimension objects."""
        self.__dimensions = OrderedDict()

    def __str__(self):
        outstr = ''
        if len(self.__dimensions) == 0:
            outstr = '<empty>'
        else:
            for kk, vv in iteritems(self.__dimensions):
                outstr += '{}: {}\n'.format(kk, vv)
        return outstr

    def __getattr__(self, name):
        # print('ATTR: {}'.format(name))
        return getattr(self.__dimensions, name)

    def __getitem__(self, item):
        """Get named dimension."""
        return self.__dimensions[item]

    @property
    def dimensions(self):
        """Get ordered dictionary of Dimension objects.

        :returns: OrderedDict of Dimension objects
        :rtype: collections.OrderedDict[str, Dimension]
        """

        return self.__dimensions

    @property
    def ndims(self):
        """Get number of dimensions.

        :returns: number of dimensions
        :rtype: int
        """

        return len(self.__dimensions)

    @property
    def xml(self):
        """Get xml element for the dimensions.

        :returns: XML element for the dimensions
        :rtype: xmlET.Element
        """

        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in iteritems(self.dimensions):
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            xmlET.SubElement(dim_sub, 'size').text = str(vv.size)
            # dim_sub.set('size', str(vv.size))
        return dims_xml

    def add(self, name, size=0):
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
                print(err)
        # else:
        #     # TODO: Should this raise an error?
        #     print('Dimension {} already exists...skipping add name'.format(name))

    def add_from_xml(self, filename):
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
        #       dictacted by the position attribute.
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

    def exists(self, name):
        """Check if dimension exists.

        :param str name: name of the dimension
        :returns: True if dimension exists, otherwise False
        :rtype: bool
        """

        return name in self.dimensions.keys()

    def get(self, name):
        """Get dimension.

        :param str name: name of the dimension

        :returns: dimension
        :rtype: Dimension

        :raises ValueError: if dimension does not exist
        """

        if self.exists(name):
            return self.__dimensions[name]
        raise ValueError('Dimension, {}, does not exist.'.format(name))

    def remove(self, name):
        """Remove dimension.

        :param str name: dimension name
        """

        if self.exists(name):
            del self.__dimensions[name]

    def tostructure(self):
        """Get data structure of Dimensions data for serialization.

        :returns: dictionary of dimension names and sizes
        :rtype: dict
        """

        dims = {}
        for kk, vv in iteritems(self.dimensions):
            dims[kk] = {'size': vv.size}
        return dims


class ParamDimensions(Dimensions):
    """Container for parameter dimensions.

    This object adds tracking of dimension position.
    """

    @property
    def xml(self):
        """Get xml for the dimensions.

        :returns: XML element of the dimensions
        :rtype: xmlET.Element
        """

        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in iteritems(self.dimensions):
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            xmlET.SubElement(dim_sub, 'position').text = str(self.get_position(kk)+1)
            xmlET.SubElement(dim_sub, 'size').text = str(vv.size)

            # dim_sub.set('position', str(self.get_position(kk)+1))
            # dim_sub.set('size', str(vv.size))
        return dims_xml

    def add_from_xml(self, filename):
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
                    raise ValueError('{}: Attempted position change from {} to {}'.format(name, curr_pos, pos))
                else:
                    if name not in ['nmonths', 'ndays', 'one']:
                        # NOTE: This will always try to grow a dimension if it already exists!
                        self.dimensions[name].size += size

    def get_position(self, name):
        """Get 0-based index position of a dimension.

        :param str name: name of the dimension

        :returns: index position of dimension
        :rtype: int
        """

        # TODO: method name should be index() ??
        return list(self.dimensions.keys()).index(name)

    def tostructure(self):
        """Get dictionary structure of the dimensions.

        :returns: dictionary of Dimensions names, sizes, and positions
        :rtype: dict
        """

        ldims = super(ParamDimensions, self).tostructure()
        for kk, vv in iteritems(ldims):
            vv['position'] = self.get_position(kk)
        return ldims
