
from typing import Any, Dict, Optional, Union   # , OrderedDict as OrderedDictType, Union

import xml.etree.ElementTree as xmlET

from .Dimension import Dimension
from ..constants import MetaDataType


class Dimensions(object):
    """Container of Dimension objects."""
    __dimensions: Dict[str, Dimension]

    def __init__(self, metadata: Optional[MetaDataType] = None,
                 verbose: Optional[bool] = False,
                 strict: Optional[bool] = True):
        """Create dictionary containing Dimension objects.

        :param verbose: Output additional debug information
        """
        self.__dimensions: Dict[str, Dimension] = {}
        self.__verbose = verbose
        self.__strict = strict
        self.metadata: Union[Dict, None] = None

        if strict:
            if metadata is None:
                raise ValueError(f'Metadata is required but was not supplied')
            self.metadata = metadata['dimensions']
        else:
            if metadata is None:
                self.metadata = {}
            else:
                # TODO: 20230707 PAN - is adhoc metadata a useful idea?
                self.metadata = metadata

        # if metadata is not None:
        #     self.metadata = metadata['dimensions']

        # if self.metadata is not None:
        #     for cdim, cvals in self.metadata.items():
        #         self.add(name=cdim, meta=self.metadata)
        #

    def __getattr__(self, name: str) -> Any:
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

    def __str__(self) -> str:
        """Pretty-print dimensions.

        :returns: Pretty-print string of dimensions
        """

        outstr = ''
        if len(self.__dimensions) == 0:
            outstr = '<empty>'
        else:
            for kk, vv in self.__dimensions.items():
                outstr += f'{vv}\n'
        return outstr

    @property
    def dimensions(self) -> Dict[str, Dimension]:
        """Get ordered dictionary of Dimension objects.

        :returns: OrderedDict of Dimension objects
        """
        return self.__dimensions

    @property
    def ndim(self) -> int:
        """Get number of dimensions.

        :returns: Number of dimensions
        """

        return len(self.__dimensions)

    @property
    def xml(self) -> xmlET.Element:
        """Get xml element for the dimensions.

        :returns: XML element for the dimensions
        """

        dims_xml = xmlET.Element('dimensions')

        for kk, vv in self.dimensions.items():
            dim_sub = xmlET.SubElement(dims_xml, 'dimension')
            dim_sub.set('name', kk)
            xmlET.SubElement(dim_sub, 'size').text = str(vv.size)
            # dim_sub.set('size', str(vv.size))
        return dims_xml

    def add(self, name: str, size: Optional[int] = None):
        """Add a new Dimension object.

        :param name: Name of the dimension
        :param size: Size of the dimension
        """

        # This method adds a dimension if it doesn't exist
        # Duplicate dimension names are silently ignored
        if name not in self.__dimensions:
            self.__dimensions[name] = Dimension(name=name, meta=self.metadata,
                                                size=size,
                                                strict=self.__strict)
        # else:
        #     # TODO: Should this raise an error?
        #     print('Dimension {} already exists...skipping add name'.format(name))

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
        else:
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

    This object adds tracking of dimension position and restricts the total number
    of individual dimensions to 2.
    """

    def __init__(self, metadata: Optional[MetaDataType] = None,
                 verbose: Optional[bool] = False,
                 strict: Optional[bool] = True):

        super(ParamDimensions, self).__init__(metadata=metadata, verbose=verbose, strict=strict)

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

    def add(self, name: str, size: Optional[int] = None):
        """Add a new Dimension object.

        :param name: Name of the dimension
        :param size: Size of the dimension
        """

        if self.ndim == 2:
            raise ValueError('A parameter cannot have more than two dimensions.')

        # Restrict number of dimensions for parameters
        super().add(name, size)

    def get_dimsize_by_index(self, index: int) -> int:
        """Return size of dimension at the given index.

        :param index: The 0-based position of the dimension
        :returns: Size of the dimension

        :raises ValueError: if index is greater than number dimensions for the parameter
        """

        if index < len(self.dimensions.items()):
            return list(self.dimensions.items())[index][1].size
        raise IndexError(f'Parameter has no dimension at index {index}')

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
