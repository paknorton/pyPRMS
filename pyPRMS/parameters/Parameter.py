import functools
import numpy as np
import numpy.typing as npt
import pandas as pd     # type: ignore
from typing import Any, cast, Dict, List, NamedTuple, Optional, Union
import xml.etree.ElementTree as xmlET

from ..constants import NEW_PTYPE_TO_DTYPE
from ..dimensions.Dimensions import ParamDimensions
from ..Exceptions_custom import FixedDimensionError

ParamDataRawType = Union[npt.NDArray, np.int32, np.float32, np.float64, np.str_]
ParamDataType = Union[npt.NDArray, np.int32, np.float32, np.float64, np.str_, int, float, str]


class Outliers(NamedTuple):
    name: str
    under: int
    over: int


class Stats(NamedTuple):
    name: str
    min: Optional[npt.DTypeLike]
    max: Optional[npt.DTypeLike]
    mean: Optional[npt.DTypeLike]
    median: Optional[npt.DTypeLike]


class Parameter(object):
    """Container for a single Parameter object.

    A parameter has a name, datatype, optional units, one or more dimensions, and
    associated data.
    """

    # Container for a single parameter
    def __init__(self, name: str,
                 meta: Optional[Dict] = None,
                 global_dims=None,
                 strict: Optional[bool] = True):
        """
        Initialize a parameter object.

        :param name: A valid PRMS parameter name
        """

        # Set the parameter name
        self.__name = name
        self.__dimensions = ParamDimensions(strict=False)

        if meta is None:
            if strict:
                raise ValueError(f'Strict is true but no metadata was supplied')
            else:
                # NOTE: Having no metadata creates a parameter with no dimensions
                self.meta = {}
        else:
            if strict:
                if name in meta:
                    self.meta = meta[name]

                    # Add the dimensions for this parameter
                    for cname in self.meta['dimensions']:
                        self.__dimensions.add(cname)

                        if global_dims is not None:
                            self.__dimensions[cname].size = global_dims.get(cname).size
                            self.__dimensions[cname].meta = global_dims[cname].meta
                else:
                    raise ValueError(f'`{self.name}` does not exist in metadata')
            else:
                # The meta must be supplied as an adhoc dictionary
                self.meta = meta

        self.__data: Optional[ParamDataRawType] = None
        self.__modified = False

    def __str__(self) -> str:
        """Pretty-print string representation of the parameter information.

        :return: Pretty-print string of arameter information
        """

        outstr = f'----- Parameter -----\n'
        outstr += f'name: {self.name}\n'

        for kk, vv in self.meta.items():
            outstr += f'{kk}: {vv}\n'

        return outstr

    @property
    def as_dataframe(self) -> pd.DataFrame:
        """Returns the parameter data as a pandas DataFrame with local model indices as the index.

        :returns: dataframe of parameter data
        """

        if len(self.data_raw.shape) == 2:
            df = pd.DataFrame(self.data_raw)
            df.rename(columns=lambda xx: '{}_{}'.format(self.name,
                                                        df.columns.get_loc(xx) + 1), inplace=True)   # type: ignore
        else:
            # Assuming 1D array
            df = pd.DataFrame(self.data_raw, columns=[self.name])   # type: ignore

        df.rename(index={k: k + 1 for k in df.index}, inplace=True)

        if self.is_hru_param():
            idx_name = 'model_hru_idx'
        elif self.is_seg_param():
            idx_name = 'model_seg_idx'
        elif self.is_poi_param():
            idx_name = 'model_poi_idx'
        else:
            idx_name = 'idx'

        df.index.name = idx_name
        return df

    @property
    def data(self) -> ParamDataType:
        """Returns the data associated with the parameter.

        :returns: parameter data
        """

        # TODO: Best way to prevent modification of data elements?
        if self.__data is not None:
            if self.is_scalar:
                return self.__data.item()
            return self.__data
        raise TypeError(f'Parameter, {self.__name}, has no data')

    @data.setter
    def data(self, data_in: ParamDataRawType):
        """Sets the data for the parameter.

        :param data_in: A list containing the parameter data
        :raises TypeError: if the datatype for the parameter is invalid
        :raises ValueError: if the number of dimensions for the parameter is greater than 2
        """

        # Metadata required: datatype, dimensions
        if self.is_scalar:
            if isinstance(data_in, np.ndarray):
                if data_in.size > 1:
                    raise IndexError(f'{self.__name}: parameter expects a scalar but '
                                     f'incoming data has size={data_in.size}')

                if data_in.dtype == NEW_PTYPE_TO_DTYPE[self.meta['datatype']]:
                    self.__data = data_in
                else:
                    # Attempt to convert to correct datatype
                    self.__data = np.array(data_in, dtype=NEW_PTYPE_TO_DTYPE[self.meta['datatype']])
            else:
                self.__data = np.array([data_in], dtype=NEW_PTYPE_TO_DTYPE[self.meta['datatype']])

            if self.__dimensions.get('one').size == 0:
                self.__dimensions.get('one').size = 1
        elif isinstance(data_in, np.ndarray) or isinstance(data_in, np.generic):
            expected_shape = tuple(ss.size for ss in self.dimensions.values())
            expected_size = functools.reduce(lambda x, y: x * y, expected_shape)

            if expected_size > 0 and data_in.shape != expected_shape:
                # If this is a parameter that was collapsed to a scalar we
                # can broadcast it to the correct shape
                if data_in.size == 1:
                    data_in = np.repeat(data_in, expected_size)

                if data_in.size == 12 and expected_shape[1] == 12:
                    # Expand nmonths to nhru, nmonth
                    data_in = np.resize(data_in, expected_shape)
                else:
                    # Try to reshape the data to match the dimensionality
                    try:
                        data_in = data_in.reshape(expected_shape, order='F')
                    except ValueError:
                        raise IndexError(f'{self.__name}: Shape of incoming data, {data_in.shape}, '
                                         f'does not match the expected shape, {expected_shape} '
                                         'and cannot be reshaped to expected shape.')

            if data_in.ndim != len(self.meta['dimensions']):
                raise IndexError(f'{self.__name}: Number of dimensions do not match '
                                 f'({data_in.ndim} != {len(self.meta["dimensions"])})')

            if data_in.dtype == NEW_PTYPE_TO_DTYPE[self.meta['datatype']]:
                self.__data = data_in
            else:
                # Attempt to convert to correct datatype
                self.__data = np.array(data_in, dtype=NEW_PTYPE_TO_DTYPE[self.meta['datatype']])

            if expected_size == 0:
                # Set the dimension size(s) if existing dimension sizes are zero
                for cname, cdim in zip(self.meta['dimensions'], self.__data.shape):
                    self.__dimensions.get(cname).size = cdim
        else:
            # TODO: 2023-11-13 PAN - This should raise an error
            pass

    @property
    def data_raw(self) -> ParamDataRawType:
        """Returns the raw data associated with the parameter.

        :returns: parameter data
        """
        if self.__data is not None:
            return self.__data
        raise TypeError(f'Parameter, {self.__name}, has no data')

    @property
    def dimensions(self) -> ParamDimensions:
        """Returns the Dimensions object associated with the parameter.

        :returns: Dimensions object for the parameter"""

        return self.__dimensions

    @property
    def index_map(self) -> Union[Dict[Any, int], None]:
        """Returns an ordered dictionary which maps data values of a 1D array
        to index positions.

        :returns: dictionary mapping data values to index position
        """

        # FIXME: 20230706 PAN - this is flawed; duplicated values overwrite
        #        index positions.
        if self.data_raw.ndim == 1:
            return dict((val.item(), idx[0]) for idx, val in np.ndenumerate(self.data_raw))
        else:
            return None

    @property
    def is_scalar(self):
        try:
            return 'one' in self.meta['dimensions']
        except KeyError:
            return True

    @property
    def modified(self) -> bool:
        """Logical denoting whether elements in the parameter data have been modified.

        :returns: True is parameter data was modified
        """
        return self.__modified

    @property
    def modules(self) -> List[str]:
        """Returns the names of the PRMS modules that require the parameter.

        :returns: names of PRMS modules that require the parameter
        """
        return self.meta.get('modules', [])

    @property
    def name(self) -> str:
        """Returns the parameter name.

        :returns: parameter name
        """
        return self.__name

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions that are defined for the parameter.

        :returns: numbers of parameter dimensions
        """
        if self.is_scalar:
            return 0
        else:
            return self.__dimensions.ndim

    # @property
    # def size(self) -> int:
    #     """Return the total size of the parameter for the defined dimensions.
    #
    #     :returns total size of parameter dimensions"""
    #     arr_shp = [dd.size for dd in self.dimensions.dimensions.values()]
    #
    #     # Compute the total size of the parameter
    #     return functools.reduce(lambda x, y: x * y, arr_shp)

    @property
    def xml(self) -> xmlET.Element:
        """Return the xml metadata for the parameter as an xml Element.

        :returns: xml element of parameter metadata
        """
        param_root = xmlET.Element('parameter')
        param_root.set('name', cast(str, self.name))
        param_root.set('version', 'ver')
        param_root.append(self.dimensions.xml)
        return param_root

    def all_equal(self) -> bool:
        """Check if all values for parameter are equal.

        :returns true if all values are equal
        """

        if self.data_raw.size > 1:
            return (self.data_raw == self.data_raw[0]).all()   # type: ignore

        return True  # scalar

    def check(self) -> str:
        """Verifies the total size of the data for the parameter matches the total declared dimension(s) size
        and returns a message.

        :returns: OK for valid data size, BAD for invalid data size
        """

        # TODO: check that values are between min and max values
        # Check a variable to see if the number of values it has is
        # consistent with the given dimensions
        if self.has_correct_size():
            # The number of values for the defined dimensions match
            return f'{self.name}: OK'
        else:
            return f'{self.name}: BAD'

    def check_values(self) -> bool:
        """Returns true if all data values are within the min/max values for the parameter.

        :returns: true when all values are within the valid min/max range for the parameter
        """
        # if self.__data is not None:
        minval = self.meta.get('minimum', None)
        maxval = self.meta.get('maximum', None)

        if minval is not None and maxval is not None:
            # Check both ends of the range
            if not (isinstance(minval, str) or isinstance(maxval, str)):
                return (self.data_raw >= minval).all() and (self.data_raw <= maxval).all().item()
            elif minval == 'bounded':
                return (self.data_raw >= self.meta.get('default')).all().item()   # type: ignore

        return True

    def has_correct_size(self) -> bool:
        """Verifies the total size of the data for the parameter matches the total declared dimension(s) sizes.

        :returns: true if size of parameter data matches declared size of dimensions
        """

        # Get the defined size for each dimension used by the variable
        total_size = 1
        for dd in self.dimensions.keys():
            total_size *= self.dimensions.get(dd).size

        return self.data_raw.size == total_size

    def is_hru_param(self) -> bool:
        """Test if parameter is dimensioned by HRU.

        :returns: true if parameter is dimensioned by nhru, ngw, or nssr
        """

        return not set(self.meta.get('dimensions', [])).isdisjoint({'nhru', 'ngw', 'nssr'})

    def is_poi_param(self) -> bool:
        """Test if parameter is dimensioned by nsegment

        :returns: true if parameter is dimensioned by npoigages
        """

        return not set(self.meta.get('dimensions', [])).isdisjoint({'npoigages'})

    def is_seg_param(self) -> bool:
        """Test if parameter is dimensioned by nsegment.

        :returns: true if parameter is dimensioned by nsegment"""

        return not set(self.meta.get('dimensions', [])).isdisjoint({'nsegment'})

    def outliers(self) -> Outliers:
        """Returns the number of values less than or greater than the valid range

        :returns: NamedTuple containing count of values less than and values greater than valid range
        """
        # Outliers = namedtuple('Outliers', ['name', 'under', 'over'])

        values_under = 0
        values_over = 0

        if self.meta.get('minimum', None) is not None:
            values_under = np.count_nonzero(self.data_raw < self.meta.get('minimum'))   # type: ignore

        if self.meta.get('maximum', None) is not None:
            values_over = np.count_nonzero(self.data_raw > self.meta.get('maximum'))   # type: ignore

        return Outliers(self.__name, values_under, values_over)

    def remove_by_index(self, dim_name: str, indices: List[int]):
        """Remove columns (nhru or nsegment) from data array given a list of indices.

        :param dim_name: Name of dimension to reduce
        :param indices: List of indices to remove"""

        if isinstance(indices, type(dict().values())):
            indices = list(indices)

        if self.__data is not None:
            if len(indices) > self.__data.size:
                raise IndexError(f'{self.name}: Cannot remove more values than exist')

            self.__data = np.delete(self.__data, indices, axis=self.dimensions.get_position(dim_name))
            assert self.__data is not None  # Needed so mypy doesn't fail on next line
            self.dimensions[dim_name].size = self.__data.shape[self.dimensions.get_position(dim_name)]
        else:
            raise TypeError('Parameter data is not initialized')

    # def reshape(self, new_dims: Dict):
    #     """Reshape a parameter, broadcasting existing values as necessary.
    #
    #     :param new_dims: Dimension names and sizes that will be used to reshape the parameter data
    #     """
    #
    #     if self.__data is None:
    #         # Reshape has no meaning if there is no data to reshape
    #         return
    #
    #     if self.dimensions.ndim == 1:
    #         if 'one' in self.dimensions.keys():
    #             # Reshaping from a scalar to a 1D or 2D array
    #             # print('Scalar to 1D or 2D')
    #             new_sizes = [vv.size for vv in new_dims.values()]
    #             tmp_data = np.broadcast_to(self.__data, new_sizes)
    #
    #             # Remove the original dimension
    #             self.dimensions.remove('one')
    #
    #             # Add the new ones
    #             for kk, vv in new_dims.items():
    #                 self.dimensions.add(kk, vv.size)
    #
    #             self.__data = tmp_data
    #         elif set(self.dimensions.keys()).issubset(set(new_dims.keys())):
    #             # Reschaping a 1D to a 2D
    #             if len(new_dims) == 1:
    #                 print('ERROR: Cannot reshape from 1D array to 1D array')
    #             else:
    #                 # print('1D array to 2D array')
    #                 new_sizes = [vv.size for vv in new_dims.values()]
    #                 try:
    #                     tmp_data = np.broadcast_to(self.__data, new_sizes)
    #                 except ValueError:
    #                     # operands could not be broadcast together with remapped shapes
    #                     tmp_data = np.broadcast_to(self.__data, new_sizes[::-1]).T
    #
    #                 old_dim = list(self.dimensions.keys())[0]
    #                 self.dimensions.remove(old_dim)
    #
    #                 for kk, vv in new_dims.items():
    #                     self.dimensions.add(kk, vv.size)
    #
    #                 self.__data = tmp_data

    def stats(self) -> Stats:
        """Returns basic statistics on parameter values.

        :returns: None (for strings or no data) or NamedTuple containing min, max, mean, and median of parameter values
        """
        # Stats = namedtuple('Stats', ['name', 'min', 'max', 'mean', 'median'])

        try:
            return Stats(self.__name, np.min(self.data_raw), np.max(self.data_raw),
                         np.mean(self.data_raw), np.median(self.data_raw))   # type: ignore
        except TypeError:
            # This happens with string data
            return Stats(self.__name, None, None, None, None)

    def subset_by_index(self, dim_name: str, indices):
        """Reduce array by axis (nhru or nsegment) a list of local indices.

        :param dim_name: name of dimension
        :param indices: local indices of HRUs or segments to extract"""

        if isinstance(indices, type(dict().values())):
            indices = list(indices)

        if self.dimensions[dim_name].is_fixed:
            raise FixedDimensionError(f'{self.name}: Cannot reduce array on a fixed dimension {dim_name}')

        # First get the index position of the given dimension name so data
        # won't be changed if the dimension name does not exist
        dim_idx = self.dimensions.get_position(dim_name)

        # We can't use the data setter when modifying the shape of parameter data
        self.__data = np.take(self.data_raw, indices, axis=dim_idx)
        assert self.data_raw is not None  # Needed so mypy doesn't fail on next line
        self.dimensions[dim_name].size = self.data_raw.shape[dim_idx]

    def tolist(self) -> List[Union[int, float, str]]:
        """Returns the parameter data as a list.

        :returns: Parameter data
        """

        # TODO: is this correct for snarea_curve?
        # Return a list of the data
        return self.data_raw.ravel(order='F').tolist()

    def toparamdb(self) -> str:
        """Outputs parameter data in the paramDb csv format.

        :returns: parameter data in the paramDb CSV format
        """

        outstr = '$id,{}\n'.format(self.name)

        ii = 0
        # Do not use self.tolist() here because it causes minor changes
        # to the values for floats.
        for dd in self.data_raw.ravel(order='F'):
            if self.meta.get('datatype', 'null') in ['float32', 'float64']:
                # Float and double types have to be formatted specially so
                # they aren't written in exponential notation or with
                # extraneous zeroes
                tmp = f'{dd:<20.7f}'.rstrip('0 ')
                if tmp[-1] == '.':
                    tmp += '0'
                outstr += f'{ii+1},{tmp}\n'
            else:
                outstr += f'{ii+1},{dd}\n'
            ii += 1
        return outstr

    def tostructure(self) -> dict:
        """Returns a dictionary structure of the parameter.

        This is typically used for serializing parameters.

        :returns: dictionary structure of the parameter
        """

        # Return all information about this parameter in the following form
        param = {'name': self.name,
                 'datatype': self.meta.get('datatype', 'null'),
                 'dimensions': self.dimensions.tostructure(),
                 'data': self.tolist()}
        return param

    def unique(self) -> Optional[npt.NDArray]:
        """Create array of unique values from the parameter data.

        :returns: Array of unique values
        """
        return np.unique(self.data_raw)

    def update_element(self, index: int, value: Union[int, float, List[int], List[float]]):
        """Update single value or row of values (e.g. nhru by nmonths) for a
        given local zero-based index in the parameter data array.

        :param index: scalar, zero-based array index
        :param value: updated value(s)
        """

        # NOTE: index is zero-based
        # Update a single element or single row (e.g. nhru x nmonth) in the
        # parameter data array.
        if self.is_scalar:
            if isinstance(value, list):
                if len(value) > 1:
                    raise TypeError(f'{self.name}: Cannot update scalar with list containing multiple values')
                value = value[0]
            elif isinstance(value, np.ndarray):
                if value.size > 1:
                    raise TypeError(f'{self.name}: Cannot update scalar with array containing multiple values')
                value = value.item()

            if self.data != value:
                # We use the data setter to make sure the new scalar is cast to a numpy array internally
                self.data = value   # type: ignore
                self.__modified = True
        else:
            if self.data_raw.ndim == 1:
                if isinstance(value, list):
                    if len(value) > 1:
                        raise TypeError(f'{self.name}: Cannot update single element with list '
                                        f'containing multiple values')
                    value = value[0]
                elif isinstance(value, np.ndarray):
                    if value.size > 1:
                        raise TypeError(f'{self.name}: Cannot update single element with array '
                                        f'containing multiple values')
                    value = value.item()
            elif self.data_raw.ndim == 2:
                if isinstance(value, list):
                    if len(value) == 1:
                        value = value[0]
                    elif len(value) != self.data_raw.shape[1]:
                        raise TypeError(f'{self.name}: Cannot update row with list of incorrect size')
                elif isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()
                    elif value.size != self.data_raw.shape[1]:
                        raise TypeError(f'{self.name}: Cannot update row with array of incorrect size')

            if not np.array_equal(self.__data[index], value):   # type: ignore
                # Change the element only if the incoming value is different
                # from the existing value
                self.__data[index] = value   # type: ignore
                self.__modified = True

    def _value_index_1d(self, value: Union[int, float, str]) -> npt.NDArray:
        """Given a scalar value return the indices where there is a match.

        :param value: The value to find in the parameter data array

        :returns: Array of zero-based indices matching the given value
        """

        if self.ndim == 1:
            # Returns a list of indices where the data elements match value
            return np.argwhere(self.data_raw == value)[:, 0]   # .tolist()
            # return np.where(self.data_raw == value)[0]
        else:
            raise TypeError(f'{self.name}: Cannot search for value in multi-dimensional array')
