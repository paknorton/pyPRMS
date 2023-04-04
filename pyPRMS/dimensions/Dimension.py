
from typing import Optional, Union  # , List

from ..constants import DIMENSION_NAMES


def _valid_dimension_name(name: str) -> bool:
    """Check if given dimension name is valid for PRMS.

    :param str name: dimension name
    :returns: True if dimension name is valid otherwise False
    """

    return name in DIMENSION_NAMES


class Dimension(object):
    """Defines a single dimension."""

    __name: str = ''
    __size: int = 0

    def __init__(self, name: str,
                 size=None,
                 meta=None):
                 # description: Optional[str] = ''):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param name: The name of the dimension
        :param size: The size of the dimension
        """

        self.__name = name

        if meta is None:
            self.meta = meta
        else:
            self.meta = meta[name]

        if size is not None:
            self.size = size
        elif self.meta is not None:
            self.size = self.meta['default']
        else:
            self.size = 0

    @property
    def name(self) -> str:
        """Name of the dimension.

        :returns: Name of the dimension
        """

        return self.__name

    @property
    def size(self) -> int:
        """Size of the dimension.

        :returns: Size of the dimension
        """

        return self.__size

    @size.setter
    def size(self, value: Union[int, str]):
        """Set the size of the dimension.

        :param value: Size of the dimension
        :raises ValueError: if dimension size is not a positive integer
        """

        value = int(value)

        if self.meta is not None:
            if self.meta['is_fixed'] and self.meta['default'] != value:
                raise ValueError(f'{self.name} is a fixed dimension and cannot be changed')

        if value < 0:
            raise ValueError('Dimension size must be a positive integer')

        self.__size = value

    def __repr__(self) -> str:
        """String respresentation of dimension.

        :returns: string with name and size of dimension
        """
        return f'Dimension(name={self.name}, size={self.size})'

    def __iadd__(self, other: int):
        """Add a number to dimension size.

        :param other: Integer value

        :returns: Dimension size

        :raises ValueError: if type of parameter is not an integer
        """

        # Augment in-place addition so the instance plus a number results
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

        # Augment in-place addition so the instance minus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        if self.__size - other < 0:
            raise ValueError('Dimension size must be positive')
        # self.size = self.__size - other
        self.size -= other
        return self
