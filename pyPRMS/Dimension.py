
from pyPRMS.constants import DIMENSION_NAMES


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

        if self.__name == 'one':
            self.__size = 1
        elif self.__name == 'nmonths':
            self.__size = 12
        elif self.__name == 'ndays':
            self.__size = 366
        else:
            self.__size = value

        if self.__name not in ['one', 'nmonths', 'ndays'] and self.__size != value:
            print('ERROR: Dimension, {}, size={}, but size {} was requested'.format(self.__name, self.__size, value))

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
