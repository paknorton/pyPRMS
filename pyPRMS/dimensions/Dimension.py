from typing import Dict, Optional, Union


class Dimension(object):
    """Defines a single dimension."""

    __name: str = ''
    __size: int = 0

    def __init__(self, name: str,
                 meta: Optional[Dict] = None,
                 size: Optional[int] = None,
                 strict: Optional[bool] = True):
        """Create a new dimension object.

        A dimension has a name and a size associated with it.

        :param name: The name of the dimension
        :param size: The size of the dimension
        :param strict: Enforce use of valid dimensions metadata
        """

        self.__name = name

        if meta is None:
            if strict:
                raise ValueError(f'Strict is true but no metadata was supplied')
            else:
                self.meta = {}
        else:
            if strict:
                if name in meta:
                    self.meta = meta[name]
                else:
                    raise ValueError(f'`{self.name}` does not exist in metadata')
            else:
                if isinstance(meta, dict):
                    # Assume we have a dictionary of metadata for this dimension
                    self.meta = meta

        if size is None:
            self.size = self.meta.get('default', 0)
        else:
            self.size = size

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
        self.size -= other
        return self

    def __repr__(self) -> str:
        """String respresentation of dimension.

        :returns: string with name and size of dimension
        """
        return f"Dimension(name='{self.name}', meta={self.meta}, size={self.size}, strict=False)"

    def __str__(self) -> str:
        """Return friendly string representation of dimension
        """
        outstr = f'----- Dimension -----\n'
        outstr += f'name: {self.name}\n'

        for kk, vv in self.meta.items():
            if kk != 'size':
                outstr += f'{kk}: {vv}\n'

        outstr += f'size: {self.size}\n'

        return outstr

    @property
    def is_fixed(self):
        return self.meta.get('is_fixed', False)

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

        if isinstance(value, float):
            raise ValueError(f'{self.name} size cannot be a float value')

        value = int(value)
        def_value = self.meta.get('default', 0)

        if value < 0:
            raise ValueError('Dimension size must be a positive integer')

        if self.is_fixed:
            if 0 < value != def_value:
                raise ValueError(f'{self.name} is a fixed dimension and cannot be changed')

            self.__size = def_value
        else:
            # The size of a dimension should never be less than the default
            if value < def_value:
                raise ValueError(f'{self.name} size cannot be less than default value ({def_value})')

            self.__size = max(value, def_value)

            # TODO: 2023-06-07 PAN - should the metadata size also get changed?
            self.meta['size'] = self.__size
