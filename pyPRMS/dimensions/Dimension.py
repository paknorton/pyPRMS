
from typing import Dict, Optional, Union  # , List

# from ..constants import DIMENSION_NAMES


# def _valid_dimension_name(name: str) -> bool:
#     """Check if given dimension name is valid for PRMS.
#
#     :param str name: dimension name
#     :returns: True if dimension name is valid otherwise False
#     """
#
#     return name in DIMENSION_NAMES


class Dimension(object):
    """Defines a single dimension."""

    __name: str = ''
    __size: int = 0

    def __init__(self, name: str,
                 meta: Optional[Dict] = None,
                 size: Optional[int] = None):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param name: The name of the dimension
        :param size: The size of the dimension
        """

        self.__name = name

        if meta is None:
            self.meta = meta
        else:
            if name not in meta:
                raise ValueError(f'`{self.name}` does not exist in metadata')

            self.meta = meta[name]

        if size is None:
            if self.meta is None:
                self.size = 0
            else:
                self.size = self.meta.get('default')
        else:
            self.size = size

    @property
    def is_fixed(self):
        if self.meta is not None:
            return self.meta.get('is_fixed', False)
        return False

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

        if value < 0:
            raise ValueError('Dimension size must be a positive integer')

        if self.meta is None:
            self.__size = value
        else:
            if self.is_fixed:
                if 0 < value != self.meta.get('default'):
                    raise ValueError(f'{self.name} is a fixed dimension and cannot be changed')

                self.__size = self.meta.get('default')
            else:
                # The size of a dimension should never be less than the default
                # TODO: 2023-05-25 PAN - should this raise an error if
                #       the incoming value is less than the default?
                if value < self.meta.get('default'):
                    raise ValueError(f'{self.name} size cannot be less than default value ({self.meta.get("default")})')

                self.__size = max(value, self.meta.get('default'))

                # TODO: 2023-06-07 PAN - should the metadata size also get changed?
                self.meta['size'] = self.__size

        # if self.meta is not None:
        #     if self.is_fixed and self.meta['default'] != value:
        #         raise ValueError(f'{self.name} is a fixed dimension and cannot be changed')

    def __repr__(self) -> str:
        """String respresentation of dimension.

        :returns: string with name and size of dimension
        """
        return f'Dimension(name={self.name}, meta={self.meta}, size={self.size})'

    def __str__(self) -> str:
        """Return friendly string representation of dimension
        """
        outstr = f'----- Dimension -----\n'
        outstr += f'name: {self.name}\n'

        if self.meta is not None:
            for kk, vv in self.meta.items():
                if kk != 'size':
                    outstr += f'{kk}: {vv}\n'

            outstr += f'size: {self.size}\n'
        else:
            outstr += f'size: {self.size}\n'
            outstr += 'No metadata for dimension\n'

        return outstr

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
        # if self.__size - other < 0:
        #     raise ValueError('Dimension size must be positive')
        self.size -= other

        return self
