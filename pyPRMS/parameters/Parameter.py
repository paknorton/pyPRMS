import functools
import numpy as np
import numpy.typing as npt   # This cannot be accessed from numpy directly
import pandas as pd     # type: ignore
from collections import namedtuple  # , OrderedDict
from typing import Any, cast, Dict, List, NamedTuple, Optional, Union
import xml.etree.ElementTree as xmlET

# from pyPRMS.Exceptions_custom import ConcatError
from ..constants import NEW_PTYPE_TO_DTYPE
from ..dimensions.Dimensions import ParamDimensions

# ParamDataType = Union[List, npt.NDArray, pd.Series, np.int32, np.float32, np.float64, np.str_]
ParamDataType = Union[npt.NDArray, np.int32, np.float32, np.float64, np.str_]


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
                else:
                    raise ValueError(f'`{self.name}` does not exist in metadata')
            else:
                # The meta must be supplied as an adhoc dictionary
                self.meta = meta

        self.__data: Optional[ParamDataType] = None
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
        """Returns the parameter data as a pandas DataFrame.

        :returns: dataframe of parameter data
        """

        if len(self.data.shape) == 2:
            df = pd.DataFrame(self.data)
            df.rename(columns=lambda xx: '{}_{}'.format(self.name, df.columns.get_loc(xx) + 1), inplace=True)
        else:
            # Assuming 1D array
            df = pd.DataFrame(self.data, columns=[self.name])

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
            return self.__data
        raise ValueError(f'Parameter, {self.__name}, has no data')

    @data.setter
    def data(self, data_in: ParamDataType):
        """Sets the data for the parameter.

        :param data_in: A list containing the parameter data
        :raises TypeError: if the datatype for the parameter is invalid
        :raises ValueError: if the number of dimensions for the parameter is greater than 2
        """

        data_np: Union[npt.NDArray, None] = None

        # Metadata required: datatype, dimensions
        if self.is_scalar:
            if isinstance(data_in, np.ndarray):
                if data_in.size > 1:
                    raise IndexError(f'{self.__name}: parameter expects a scalar but incoming data has size={data_in.size}')
                if data_in.dtype == NEW_PTYPE_TO_DTYPE[self.meta['datatype']]:
                    self.__data = data_in
                else:
                    # Attempt to convert to correct datatype
                    self.__data = np.array(data_in, dtype=NEW_PTYPE_TO_DTYPE[self.meta['datatype']])
            else:
                self.__data = NEW_PTYPE_TO_DTYPE[self.meta['datatype']](data_in)

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
                raise IndexError(f'{self.__name}: Number of dimensions do not match ({data_in.ndim} != {len(self.meta["dimensions"])})')

            # if self.__data is not None:
            #     # Make sure shapes match
            #     if data_in.shape != self.__data.shape:
            #         raise IndexError(f'{self.__name}: Shape of incoming data, {data_in.shape}, '
            #                          f'does not match shape of existing data, {self.__data.shape}')

            if data_in.dtype == NEW_PTYPE_TO_DTYPE[self.meta['datatype']]:
                self.__data = data_in
            else:
                # Attempt to convert to correct datatype
                self.__data = np.array(data_in, dtype=NEW_PTYPE_TO_DTYPE[self.meta['datatype']])
                # raise ValueError(f'{self.__name}: incoming datatype, {data_in.dtype}, does not match expected, {NEW_PTYPE_TO_DTYPE[self.meta["datatype"]]}')

            if expected_size == 0:
                # Set the dimension size(s) if existing dimension sizes are zero
                for cname, cdim in zip(self.meta['dimensions'], self.__data.shape):
                    self.__dimensions.get(cname).size = cdim
        else:
            # TODO: 2023-11-13 PAN - This should raise an error
            pass

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
        if isinstance(self.__data, np.ndarray) and self.__data.ndim == 1:
            return dict((val, idx) for idx, val in enumerate(self.__data.tolist()))
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
        # if self.__data is not None:
        if self.data.size > 1:
            return (self.__data == self.__data[0]).all()   # type: ignore
        return True  # scalar
        # else:
        #     raise TypeError('Parameter data is not initialized')

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
            if not(isinstance(minval, str) or isinstance(maxval, str)):
                return bool((self.data >= minval).all() and (self.data <= maxval).all())
            elif minval == 'bounded':
                return bool((self.data >= self.meta.get('default')).all())
        return True
        # else:
        #     raise TypeError('Parameter data is not initialized')

    def has_correct_size(self) -> bool:
        """Verifies the total size of the data for the parameter matches the total declared dimension(s) sizes.

        :returns: true if size of parameter data matches declared size of dimensions
        """

        # Get the defined size for each dimension used by the variable
        total_size = 1
        for dd in self.dimensions.keys():
            total_size *= self.dimensions.get(dd).size

        return self.data.size == total_size

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

    def outliers(self) -> NamedTuple:
        """Returns the number of values less than or greater than the valid range

        :returns: NamedTuple containing count of values less than and values greater than valid range
        """
        Outliers = namedtuple('Outliers', ['name', 'under', 'over'])

        values_under = 0
        values_over = 0

        if self.__data is not None:
            if self.meta.get('minimum', None) is not None:
                values_under = np.count_nonzero(self.__data < self.meta.get('minimum'))
                # values_under = len(self.__data[self.__data < self.__minimum])

            if self.meta.get('maximum', None) is not None:
                values_over = np.count_nonzero(self.__data > self.meta.get('maximum'))
                # values_over = len(self.__data[self.__data > self.__maximum])

        return Outliers(self.__name, values_under, values_over)

    def remove_by_index(self, dim_name: str, indices: List[int]):
        """Remove columns (nhru or nsegment) from data array given a list of indices.

        :param dim_name: Name of dimension to reduce
        :param indices: List of indices to remove"""

        if isinstance(indices, type(Dict().values())):
            indices = list(indices)

        if self.__data is not None:
            if self.__data.size == 1:
                print(f'{self.name}: Cannot reduce array of size one')
                return

            self.__data = np.delete(self.__data, indices, axis=self.dimensions.get_position(dim_name))
            assert self.__data is not None  # Needed so mypy doesn't fail on next line
            self.dimensions[dim_name].size = self.__data.shape[self.dimensions.get_position(dim_name)]
        else:
            raise TypeError('Parameter data is not initialized')

    def reshape(self, new_dims: Dict):
        """Reshape a parameter, broadcasting existing values as necessary.

        :param new_dims: Dimension names and sizes that will be used to reshape the parameter data
        """

        if self.__data is None:
            # Reshape has no meaning if there is no data to reshape
            return

        if self.dimensions.ndim == 1:
            if 'one' in self.dimensions.keys():
                # Reshaping from a scalar to a 1D or 2D array
                # print('Scalar to 1D or 2D')
                new_sizes = [vv.size for vv in new_dims.values()]
                tmp_data = np.broadcast_to(self.__data, new_sizes)

                # Remove the original dimension
                self.dimensions.remove('one')

                # Add the new ones
                for kk, vv in new_dims.items():
                    self.dimensions.add(kk, vv.size)

                self.__data = tmp_data
            elif set(self.dimensions.keys()).issubset(set(new_dims.keys())):
                # Reschaping a 1D to a 2D
                if len(new_dims) == 1:
                    print('ERROR: Cannot reshape from 1D array to 1D array')
                else:
                    # print('1D array to 2D array')
                    new_sizes = [vv.size for vv in new_dims.values()]
                    try:
                        tmp_data = np.broadcast_to(self.__data, new_sizes)
                    except ValueError:
                        # operands could not be broadcast together with remapped shapes
                        tmp_data = np.broadcast_to(self.__data, new_sizes[::-1]).T

                    old_dim = list(self.dimensions.keys())[0]
                    self.dimensions.remove(old_dim)

                    for kk, vv in new_dims.items():
                        self.dimensions.add(kk, vv.size)

                    self.__data = tmp_data

    def stats(self) -> Optional[NamedTuple]:
        """Returns basic statistics on parameter values.

        :returns: None (for strings or no data) or NamedTuple containing min, max, mean, and median of parameter values
        """
        Stats = namedtuple('Stats', ['name', 'min', 'max', 'mean', 'median'])

        # if self.__name in ['poi_gage_id']:
        #     return Stats(self.__name, '', '', '', '')

        if self.__data is None:
            return None

        try:
            return Stats(self.__name, np.min(self.__data), np.max(self.__data),
                         np.mean(self.__data), np.median(self.__data))
        except TypeError:
            # This happens with string data
            return None

    def subset_by_index(self, dim_name: str, indices):
        """Reduce columns (nhru or nsegment) from data array given a list of indices.

        :param dim_name: name of dimension
        :param indices: indices of HRUs or segments to extract"""

        if isinstance(indices, type(Dict().values())):
            indices = list(indices)

        if self.__data is not None:
            if self.__data.size == 1:
                print(f'{self.name}: Cannot reduce array of size one')
                return

            self.__data = self.__data[indices]
            assert self.__data is not None  # Needed so mypy doesn't fail on next line
            self.dimensions[dim_name].size = self.__data.shape[self.dimensions.get_position(dim_name)]
            # self.__data = np.take(self.__data, indices, axis=0)
            # self.__data = np.delete(self.__data, indices, axis=self.dimensions.get_position(dim_name))
        else:
            raise TypeError('Parameter data is not initialized')

    def tolist(self) -> List[Union[int, float, str]]:
        """Returns the parameter data as a list.

        :returns: Parameter data
        """

        # TODO: is this correct for snarea_curve?
        # Return a list of the data
        if self.__data is not None:
            return self.__data.ravel(order='F').tolist()
        else:
            raise TypeError('Parameter data is not initialized')

    def toparamdb(self) -> str:
        """Outputs parameter data in the paramDb csv format.

        :returns: parameter data in the paramDb CSV format
        """

        if self.__data is not None:
            outstr = '$id,{}\n'.format(self.name)

            ii = 0
            # Do not use self.tolist() here because it causes minor changes
            # to the values for floats.
            for dd in self.__data.ravel(order='F'):
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
        else:
            raise TypeError('Parameter data and/or metadata is not initialized')

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
        if self.__data is None:
            return None

        return np.unique(self.__data)

    def update_element(self, index: int, value: Union[int, float, List[int], List[float]]):
        """Update single value or row of values (e.g. nhru by nmonths) for a
        given zero-based index in the parameter data array.

        :param index: scalar, zero-based array index
        :param value: updated value(s)
        """

        # NOTE: index is zero-based
        # Update a single element or single row (e.g. nhru x nmonth) in the
        # parameter data array.
        if self.__data is not None:
            if self.is_scalar:
                if self.__data != value:
                    self.__data = value   # type: ignore
                    self.__modified = True
            else:
                if not np.array_equal(self.__data[index], value):   # type: ignore
                    # Change the element only if the incoming value is different
                    # from the existing value
                    self.__data[index] = value   # type: ignore
                    self.__modified = True
        else:
            raise TypeError('Parameter data is not initialized')

    def _value_index(self, value: Union[int, float, str]) -> Union[npt.NDArray, None]:
        """Given a scalar value return the indices where there is a match.

        :param value: The value to find in the parameter data array

        :returns: Array of zero-based indices matching the given value
        """

        if self.ndim > 1:
            # TODO: 2021-03-24 PAN - add support for 2D arrays
            print(f'{self.name}: _value_index() does not support 2D arrays yet')
            return None
        else:
            # Returns a list of indices where the data elements match value
            return np.where(self.__data == value)[0]

    # def concat(self, data_in):
    #     """Takes a list of parameter data and concatenates it to the end of the existing parameter data.
    #
    #     This is useful when reading 2D parameter data by region where
    #     the ordering of the data must be correctly maintained in the final
    #     dataset
    #
    #     :param list data_in: Data to concatenate (or append) to existing parameter data
    #     :raises TypeError: if the datatype for the parameter is invalid
    #     :raises ValueError: if the number of dimensions for the parameter is greater than 2
    #     :raises ConcatError: if concatenation is attempted with a parameter of dimension 'one' (e.g. scalar)
    #     """
    #
    #     if not self.ndim:
    #         raise ValueError(f'No dimensions have been defined for {self.name}. Unable to concatenate data')
    #
    #     if self.__data is None:
    #         # Don't bother with the concatenation if there is no pre-existing data
    #         self.data = data_in
    #         return
    #
    #     # Convert datatype first
    #     datatype_conv = {1: self.__str_to_int, 2: self.__str_to_float,
    #                      3: self.__str_to_float, 4: self.__str_to_str}
    #
    #     if self.__datatype in DATA_TYPES.keys():
    #         data_in = datatype_conv[self.__datatype](data_in)
    #     else:
    #         raise TypeError(f'Defined datatype {self.__datatype} for parameter {self.__name} is not valid')
    #
    #     # Convert list to np.array
    #     if self.ndim == 2:
    #         data_np = np.array(data_in).reshape((-1, self.dimensions.get_dimsize_by_index(1),), order='F')
    #     elif self.ndim == 1:
    #         data_np = np.array(data_in)
    #     else:
    #         raise ValueError(f'Number of dimensions, {self.ndim}, is not supported')
    #
    #     if 'one' in self.__dimensions.dimensions.keys():
    #         # A parameter with the dimension 'one' should never have more
    #         # than 1 value. Output warning if the incoming value is different
    #         # from a pre-existing value
    #         if data_np[0] != self.__data[0]:
    #             raise ConcatError(f'Parameter, {self.__name}, with dimension "one" already ' +
    #                               f'has assigned value = {self.__data[0]}; ' +
    #                               f'Cannot concatenate additional value(s), {data_np[0]}')
    #             # print('WARNING: {} with dimension "one" has different '.format(self.__name) +
    #             #       'value ({}) from current ({}). Keeping current value.'.format(data_np[0], self.__data[0]))
    #     else:
    #         self.__data = np.concatenate((self.__data, data_np))
    #         # self.__data = data_np
