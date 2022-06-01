
from typing import Optional, Union, List

from pyPRMS.constants import DIMENSION_NAMES


def _valid_dimension_name(name: str) -> bool:
    """Check if given dimension name is valid for PRMS.

    :param str name: dimension name
    :returns: True if dimension name is valid otherwise False
    :rtype: bool
    """

    return name in DIMENSION_NAMES


class Dimension(object):
    """Defines a single dimension."""
    __name: str = ''
    __size: int = 0
    # __description: Optional[str] = None

    # def __init__(self, name: Optional[str],
    #              description: Optional[str],
    #              size: Optional[int] = 0):
    def __init__(self, name: str,
                 size: int=0,
                 description: Optional[str]=''):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param name: The name of the dimension
        :param size: The size of the dimension
        :param description: Description of the dimension
        """

        self.name = name
        self.size = size

        if description is not None:
            self.description = description

    @property
    def name(self) -> str:
        """Name of the dimension.

        :returns: Name of the dimension
        """

        return self.__name

    @name.setter
    def name(self, name: str):
        """Sets the name of the dimension.

        :param name: Name of the dimension
        :raises ValueError: if dimension name is not a valid PRMS dimension
        """

        if _valid_dimension_name(name):
            self.__name = name
        else:
            # TODO: Should this always raise an error?
            raise ValueError(f'Dimension name, {name}, is not a valid PRMS dimension name')

    @property
    def size(self) -> int:
        """Size of the dimension.

        :returns: Size of the dimension
        """

        return self.__size

    @size.setter
    def size(self, value: Union[int, str]) -> None:
        """Set the size of the dimension.

        :param value: Size of the dimension
        :raises ValueError: if dimension size in not a positive integer
        """
        if isinstance(value, str):
            value = int(value)

        if not isinstance(value, int) or value < 0:
            raise ValueError('Dimension size must be a positive integer')

        if self.__name == 'one':
            if value != 1:
                raise ValueError('Dimension named "one" must have size=1')
            # assert value == 1, 'Dimension, one, must have size=1'
            self.__size = 1
        elif self.__name == 'nmonths':
            if value != 12:
                raise ValueError('Dimension named "nmonths" must have size=12')
            # assert value == 12, 'Dimension, nmonths, must have size=12'
            self.__size = 12
        elif self.__name == 'ndays':
            if value != 366:
                raise ValueError('Dimension named "ndays" must have size=366')
            self.__size = 366
        else:
            self.__size = value

        if self.__name not in ['one', 'nmonths', 'ndays'] and self.__size != value:
            print(f'ERROR: Dimension, {self.__name}, size={self.__size}, but size {value} was requested')

    @property
    def description(self) -> str:
        """Description for the dimension.

        :returns: Description for the dimension
        """

        return self.__description

    @description.setter
    def description(self, descstr: str):
        """Set the description of the dimension.

        :param descstr: Description string
        """

        self.__description = descstr

    def __repr__(self) -> str:
        return 'Dimension(name={}, size={!r})'.format(self.name, self.size)

    def __iadd__(self, other: int):
        """Add a number to dimension size.

        :param other: Integer value

        :returns: Dimension size

        :raises ValueError: if type of parameter is not an integer
        """

        # augment in-place addition so the instance plus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        self.size += other
        return self

    def __isub__(self, other: int):
        """Subtracts integer from dimension size.

        :param other: Integer value

        :returns: Dimension size

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
