
from __future__ import (absolute_import, division, print_function)
from future.utils import iteritems

from collections import OrderedDict
import xml.etree.ElementTree as xmlET

from pyPRMS.constants import DIMENSION_NAMES
from pyPRMS.prms_helpers import read_xml


def _valid_dimension_name(name):
    """Returns true if given dimension name is a valid name for PRMS

    :param name: dimension name
    :returns: boolean (True if dimension name is valid otherwise False)
    """

    return name in DIMENSION_NAMES


class Dimension(object):
    """Defines a single dimension"""

    # Container for a single dimension
    def __init__(self, name=None, size=0, description=None):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param name: The name of the dimension
        :param size: The size of the dimension
        """

        self.__name = None
        self.__size = None
        self.__description = None
        self.name = name  # Name of the dimension
        self.size = size  # integer
        self.description = description

    @property
    def name(self):
        """Returns the name of the dimension"""
        return self.__name

    @name.setter
    def name(self, name):
        """Sets the name of the dimension

        :param name: The name of the dimension
        """

        if _valid_dimension_name(name):
            self.__name = name
        else:
            # TODO: Should this raise an error?
            raise ValueError('Dimension name, {}, is not a valid PRMS dimension name'.format(name))

    @property
    def size(self):
        """"Returns the size of the dimension"""
        return self.__size

    @size.setter
    def size(self, value):
        """Set the size of the dimension

        :param value: The total size of the dimension"""
        if not isinstance(value, int) or value < 0:
            raise ValueError('Dimension size must be a positive integer')
        self.__size = value

    @property
    def description(self):
        """"Returns the description for the dimension"""
        return self.__description

    @description.setter
    def description(self, descstr):
        """Set the description of the dimension

        :param descstr: Description string
        """
        self.__description = descstr

    def __repr__(self):
        return 'Dimension(name={}, size={!r})'.format(self.name, self.size)

    def __iadd__(self, other):
        """Adds integer to dimension size

        :param other: Integer value
        :returns: integer dimension size.
        """

        # augment in-place addition so the instance plus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        self.__size += other
        return self

    def __isub__(self, other):
        """Subtracts integer from dimension size

        :param other: Integer value
        :returns: integer dimension size
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
    """Container of Dimension objects"""

    def __init__(self):
        """Create ordered dictionary to contain Dimension objects"""
        self.__dimensions = OrderedDict()  # ordered dictionary of Dimension()

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
        """Get named dimension"""
        return self.__dimensions[item]

    @property
    def dimensions(self):
        """Returns ordered dictionary of Dimension objects"""
        # Return the ordered dictionary of defined dimensions
        return self.__dimensions

    @property
    def ndims(self):
        """Return the total number of dimensions"""
        # Number of dimensions
        return len(self.__dimensions)

    @property
    def xml(self):
        """Returns the xml for the dimensions"""
        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in iteritems(self.dimensions):
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            dim_sub.set('size', str(vv.size))
        return dims_xml

    def add(self, name, size=0):
        """Add a new dimension.

        :param name: The name of the dimension.
        :param size: The size of the dimension.
        """

        # This method adds a dimension if it doesn't exist
        # TODO: check for valid dimension size for ndays, nmonths, and one
        if name not in self.__dimensions:
            try:
                self.__dimensions[name] = Dimension(name=name, size=size)
            except ValueError as err:
                print(err)
        else:
            # TODO: Should this raise an error?
            print('Dimension {} already exists...skipping add name'.format(name))

    def add_from_xml(self, filename):
        """Add one or more dimensions from an xml file

        :param filename: The name of the xml file to read.
        """

        # Add dimensions and grow dimension sizes from xml information for a parameter
        # This information is found in xml files for each region for each parameter
        # No attempt is made to verify whether each region for a given parameter
        # has the same or same number of dimensions.
        xml_root = read_xml(filename)

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
        """Verifies if a dimension exists

        :param name: The name of the dimension
        :returns: boolen (True if dimension exists otherwise False).
        """

        return name in self.dimensions.keys()

    def get(self, name):
        """Returns the given dimensions if it exists

        :param name: name of the dimension
        :returns: dimension object
        """

        if self.exists(name):
            return self.__dimensions[name]
        raise ValueError('Dimension, {}, does not exist.'.format(name))

    def remove(self, name):
        """Removes a dimension

        :param name: dimension name
        """

        if self.exists(name):
            del self.__dimensions[name]

    def tostructure(self):
        """Returns a data structure of Dimensions data for serialization"""
        # Return the dimensions info/data as a data structure
        dims = {}
        for kk, vv in iteritems(self.dimensions):
            dims[kk] = {'size': vv.size}
        return dims


class ParamDimensions(Dimensions):
    """Container for parameter dimensions. This object adds tracking dimension position."""

    @property
    def xml(self):
        """Returns the xml for the dimensions"""
        # <dimensions>
        #     <dimension name = "nsegment" position = "1" size = "1434" />
        # </ dimensions>
        dims_xml = xmlET.Element('dimensions')

        for kk, vv in iteritems(self.dimensions):
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            dim_sub.set('position', str(self.get_position(kk)+1))
            dim_sub.set('size', str(vv.size))
        return dims_xml

    def add_from_xml(self, filename):
        """Add one or more dimensions from an xml file. This version also checks dimension position.

        :param filename: The name of the xml file to read.
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
                curr_pos = self.dimensions.keys().index(name)

                if curr_pos != pos:
                    # This indicates a problem in one of the paramdb files
                    raise ValueError('{}: Attempted position change from {} to {}'.format(name, curr_pos, pos))
                else:
                    if name not in ['nmonths', 'ndays', 'one']:
                        # NOTE: This will always try to grow a dimension if it already exists!
                        self.dimensions[name].size += size

    def get_position(self, name):
        """Returns the 0-based index position of the given dimension name"""
        # TODO: method name should be index() ??
        return self.dimensions.keys().index(name)

    def tostructure(self):
        """Returns a structure of the dimensions including position information"""
        ldims = super(ParamDimensions, self).tostructure()
        for kk, vv in iteritems(ldims):
            vv['position'] = self.get_position(kk)
        return ldims
