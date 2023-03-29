import functools
import numpy as np
import numpy.typing as npt   # This cannot be accessed from numpy directly
import pandas as pd     # type: ignore
from collections import namedtuple, OrderedDict
from typing import Any, cast, NamedTuple, Optional, Union, List
import xml.etree.ElementTree as xmlET

# from pyPRMS.Exceptions_custom import ConcatError
from ..constants import DATA_TYPES, DATATYPE_TO_DTYPE
from ..dimensions.Dimensions import ParamDimensions


class Parameter(object):
    """Container for a single Parameter object.

    A parameter has a name, datatype, optional units, one or more dimensions, and
    associated data.
    """

    # Container for a single parameter
    # TODO: 2021-12-03 PAN - The arguments should not all be optional
    def __init__(self, name: str,
                 datatype: Optional[int]=None,
                 units: Optional[str]=None,
                 model: Optional[str]=None,
                 description: Optional[str]=None,
                 help: Optional[str]=None,
                 modules: Optional[Union[str, List[str]]]=None,
                 minimum: Optional[Union[int, float, str]]=None,
                 maximum: Optional[Union[int, float, str]]=None,
                 default: Optional[Union[int, float, str]]=None):
        """
        Initialize a parameter object.

        :param name: A valid PRMS parameter name
        :param datatype: The datatype for the parameter (1-Integer, 2-Float, 3-Double, 4-String)
        :param units: Optional units string for the parameter
        :param model: Model string for parameter
        :param description: Description of the parameter
        :param help: Help text for the parameter
        :param modules: List of modules that require the parameter
        :param minimum: Minimum value allowed in the parameter data
        :param maximum: Maximum value allowed in the parameter data
        :param default: Default value used for parameter data
        """

        # Set the parameter name
        self.__name = name

        # Initialize internal variables
        # self.__datatype = None
        self.__units = ''
        self.__model = ''
        self.__description = ''
        self.__help = ''
        self.__modules: Optional[List[str]] = None
        self.__minimum: Optional[Union[int, float, str]] = None
        self.__maximum: Optional[Union[int, float, str]] = None
        self.__default: Optional[Union[int, float, str]] = None

        self.__dimensions = ParamDimensions()
        self.__data: Optional[npt.NDArray] = None  # array
        # self.__data: Optional[npt.NDArray[Union[np.int_, np.float_, np.str_]]] = None  # array

        self.__modified = False

        # Use setters for most internal variables
        self.datatype = datatype    # type: ignore
        self.units = units  # type: ignore
        self.model = model  # type: ignore
        self.description = description  # type: ignore
        self.help = help    # type: ignore
        self.modules = modules  # type: ignore
        self.minimum = minimum  # type: ignore
        self.maximum = maximum  # type: ignore
        self.default = default  # type: ignore

    def __str__(self) -> str:
        """Pretty-print string representation of the parameter information.

        :return: Pretty-print string of arameter information
        """
        out_text = 'name: {}\ndatatype: {}\nunits: {}\nndims: {}\ndescription: {}\nhelp: {}\n'
        outstr = out_text.format(self.name, self.datatype, self.units, self.ndims, self.description,
                                 self.help)

        if self.__minimum is not None:
            outstr += f'Minimum value: {self.__minimum}\n'

        if self.__maximum is not None:
            outstr += f'Maximum value: {self.__maximum}\n'

        if self.__default is not None:
            outstr += f'Default value: {self.__default}\n'

        outstr += 'Size of data: '
        if self.__data is not None:
            outstr += f'{self.data.size}\n'
        else:
            outstr += '<empty>\n'

        if self.__modules is not None:
            outstr += 'Modules: '

            for xx in self.__modules:
                outstr += f'{xx} '
            outstr += '\n'

        if self.ndims:
            outstr += 'Dimensions:\n' + self.__dimensions.__str__()
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
    def data(self) -> npt.NDArray:
        """Returns the data associated with the parameter.

        :returns: parameter data
        """

        # TODO: Best way to prevent modification of data elements?
        if self.__data is not None:
            return self.__data
        raise ValueError(f'Parameter, {self.__name}, has no data')

    @data.setter
    def data(self, data_in: Union[List, npt.NDArray, pd.Series]):
        """Sets the data for the parameter.

        :param data_in: A list containing the parameter data
        :raises TypeError: if the datatype for the parameter is invalid
        :raises ValueError: if the number of dimensions for the parameter is greater than 2
        """
        # Raise an error if no dimensions are defined for parameter
        if not self.ndims:
            raise ValueError(f'No dimensions have been defined for {self.name}; unable to append data')

        data_np: Union[npt.NDArray, None] = None

        if isinstance(data_in, list):
            data_np = np.array(data_in, dtype=DATATYPE_TO_DTYPE[self.datatype])
        elif isinstance(data_in, np.ndarray):
            data_np = data_in
        elif isinstance(data_in, pd.Series):
            data_np = data_in.to_numpy()
        else:
            raise TypeError('Right-hand variable not of type list, ndarray, or Pandas Series')

        assert data_np is not None

        if data_np.size == self.size:
            # The incoming size matches the expected size for the parameter
            if data_np.ndim < self.ndims:
                # A numpy scalar (size=1, ndim=0) should not be reshaped

                if data_np.ndim != 0:
                    # Assume data_np is 1D, parameter is 2D; there are no scalars
                    # order higher dimension possibilities.
                    data_np = data_np.reshape((-1, self.dimensions.get_dimsize_by_index(1),), order='F')
            elif data_np.ndim == self.ndims:
                # TODO: If dealing with 2D incoming data should make sure that
                #       the shape is correct compared to declared dimensions.
                pass
            else:
                raise ValueError(f'{self.__name}, source data ndim, {data_np.ndim} > parameter ndim, {self.ndims}')
        elif data_np.size > self.size and 'one' in self.__dimensions.dimensions.keys():
            # In certain circumstances it is possible for a one-dimensioned
            # parameter to be passed a data array with size > 1. If this happens
            # just use the first element from the array.
            print(f'WARNING: {self.__name}, with dimension "one" was passed {data_np.size} ' +
                  f'values; using first value only.')
            data_np = np.array(data_np[0], ndmin=1)
        else:
            if data_np.size == 1:
                # Incoming data is scalar but it should be an array; expand to the expected dims/size
                new_sizes = [vv.size for vv in self.dimensions.values()]
                data_np = np.broadcast_to(data_np, new_sizes)
                print(f'{self.name}: Scalar was broadcast to {new_sizes}')
            else:
                print(f'{self.size=}; {data_np.size=}')
                err_txt = f'{self.name}: Number of dimensions for new data ({data_np.ndim}) ' + \
                          f'doesn\'t match old ({self.ndims})'
                raise IndexError(err_txt)

        if self.__data is None:
            self.__data = data_np
        elif np.array_equal(self.__data, data_np):
            pass
            # print(f'{self.__name}: updated value is equal to the old value')
        else:
            # Pre-existing data has been modified
            self.__data = data_np
            self.__modified = True

    @property
    def datatype(self) -> int:
        """Returns the datatype of the parameter.

        :returns: datatype of the parameter data
        """
        return self.__datatype

    @datatype.setter
    def datatype(self, dtype: int):
        """Sets the datatype for the parameter.

        :param dtype: The datatype for the parameter (1-Integer, 2-Float, 3-Double, 4-String)
        """

        # TODO: Should this be able to handle both string (e.g. 'I') and integer datatypes?
        # TODO: If datatype is changed should verify existing data can be cast to it
        if dtype in DATA_TYPES:
            self.__datatype = dtype
        elif dtype is None:
            self.__datatype = None
        else:
            # TODO: This should raise and error (what kind?)
            raise TypeError(f'Invalid datatype, {dtype}, specified for parameter')

    @property
    def default(self) -> Union[int, float, str, None]:
        """Returns the default value for the parameter.

        :returns: Default value defined for the parameter
        """
        return self.__default

    @default.setter
    def default(self, value: Union[int, float, str, None]):
        """Set the default value for the parameter.

        :param value: default value for the parameter
        """
        # TODO: 2020-04-30 PAN: This should check if given value is between
        #                       min and max valid values (if set)
        if self.__datatype is None or value is None:
            self.__default = value
        elif DATA_TYPES[self.__datatype] == 'float':
            self.__default = float(value)
        elif DATA_TYPES[self.__datatype] == 'integer':
            self.__default = int(value)
        else:
            self.__default = value

    @property
    def description(self) -> str:
        """Returns the parameter description.

        :returns: parameter description
        """
        return self.__description

    @description.setter
    def description(self, descstr: str):
        """Set the model description for the parameter.

        :param descstr: Description string
        """
        self.__description = descstr

    @property
    def dimensions(self) -> ParamDimensions:
        """Returns the Dimensions object associated with the parameter.

        :returns: Dimensions object for the parameter"""
        return self.__dimensions

    @property
    def help(self) -> str:
        """Returns the help information for the parameter.

        :returns: parameter help information
        """
        return self.__help

    @help.setter
    def help(self, helpstr: str):
        """Set the help string for the parameter.

        :param helpstr: Help string
        """
        self.__help = helpstr

    @property
    def index_map(self) -> Union[OrderedDict[Any, int], None]:
        """Returns an ordered dictionary which maps data values to index position.

        :returns: dictionary mapping data values to index position
        """

        if self.__data is None:
            return None

        return OrderedDict((val, idx) for idx, val in enumerate(self.__data.tolist()))

    @property
    def maximum(self) -> Union[int, float, str, None]:
        """Returns the maximum valid value for the parameter.

        :returns: maximum valid data value
        """
        return self.__maximum

    @maximum.setter
    def maximum(self, value: Union[int, float, str, None]):
        """Set the maximum valid value for the parameter.

        :param value: The maximum valid value
        """
        if self.__datatype is None or value is None:
            self.__maximum = value
        elif DATA_TYPES[self.__datatype] == 'float':
            self.__maximum = float(value)
        elif DATA_TYPES[self.__datatype] == 'integer':
            try:
                self.__maximum = int(value)
            except ValueError:
                # This happens with bounded parameters
                self.__maximum = value
        else:
            self.__maximum = value

    @property
    def minimum(self) -> Union[int, float, str, None]:
        """Returns the minimum valid value for the parameter.

        :returns: minimum valid data value
        """
        return self.__minimum

    @minimum.setter
    def minimum(self, value: Union[int, float, str, None]):
        """Set the minimum valid value for the parameter.

        :param value: The minimum valid value
        """
        if self.__datatype is None or value is None:
            self.__minimum = value
        elif DATA_TYPES[self.__datatype] == 'float':
            self.__minimum = float(value)
        elif DATA_TYPES[self.__datatype] == 'integer':
            try:
                self.__minimum = int(value)
            except ValueError:
                # This happens with 'bounded' parameters
                self.__minimum = str(value)
        else:
            self.__minimum = value

    @property
    def model(self) -> str:
        """Returns the model the parameter is used in.

        :returns: model string
        """
        return self.__model

    @model.setter
    def model(self, modelstr: str):
        """Set the model description for the parameter.

        :param modelstr: String denoting the model (e.g. PRMS)
        """
        self.__model = modelstr

    @property
    def modified(self) -> bool:
        """Logical denoting whether parameter data has been modified.

        :returns: True is parameter data was modified
        """
        return self.__modified

    @property
    def modules(self) -> Union[List[str], None]:
        """Returns the names of the PRMS modules that require the parameter.

        :returns: names of PRMS modules that require the parameter
        """
        return self.__modules

    @modules.setter
    def modules(self, modulestr: Union[str, List[str], None]):
        """Set the names of the modules that require the parameter.

        :param modulestr: Single module name or list of module names to add
        """
        if modulestr is not None:
            if isinstance(modulestr, list):
                self.__modules = modulestr
            else:
                self.__modules = [modulestr]
        else:
            self.__modules = None

    @property
    def name(self) -> str:
        """Returns the parameter name.

        :returns: parameter name
        """
        return self.__name

    @property
    def ndims(self) -> int:
        """Returns the number of dimensions that are defined for the parameter.

        :returns: numbers of parameter dimensions
        """
        return self.__dimensions.ndims

    @property
    def size(self) -> int:
        """Return the total size of the parameter for the defined dimensions.

        :returns total size of parameter dimensions"""
        arr_shp = [dd.size for dd in self.dimensions.dimensions.values()]

        # Compute the total size of the parameter
        return functools.reduce(lambda x, y: x * y, arr_shp)

    @property
    def units(self) -> str:
        """Returns the parameter units.

        :returns: units for the parameter
        """
        return self.__units

    @units.setter
    def units(self, unitstr: str):
        """Set the parameter units.

        :param unitstr: String denoting the units for the parameter (e.g. mm)
        """
        self.__units = unitstr

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
        if self.__data is not None:
            if self.__data.size > 1:
                return (self.__data == self.__data[0]).all()
            return False
        else:
            raise TypeError('Parameter data is not initialized')

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
        if self.__data is not None:
            if self.__minimum is not None and self.__maximum is not None:
                # Check both ends of the range
                if not(isinstance(self.__minimum, str) or isinstance(self.__maximum, str)):
                    return bool((self.__data >= self.__minimum).all() and (self.__data <= self.__maximum).all())
                elif self.__minimum == 'bounded':
                    return bool((self.__data >= self.__default).all())
            return True
        else:
            raise TypeError('Parameter data is not initialized')

    def has_correct_size(self) -> bool:
        """Verifies the total size of the data for the parameter matches the total declared dimension(s) sizes.

        :returns: true if size of parameter data matches declared size of dimensions
        """

        # Get the defined size for each dimension used by the variable
        total_size = 1
        for dd in self.dimensions.keys():
            total_size *= self.dimensions.get(dd).size

        # This assumes a numpy array
        return self.data.size == total_size

    def is_hru_param(self) -> bool:
        """Test if parameter is dimensioned by HRU.

        :returns: true if parameter is dimensioned by nhru, ngw, or nssr
        """

        return not set(self.__dimensions.keys()).isdisjoint({'nhru', 'ngw', 'nssr'})

    def is_poi_param(self) -> bool:
        """Test if parameter is dimensioned by nsegment

        :returns: true if parameter is dimensioned by npoigages
        """

        return not set(self.__dimensions.keys()).isdisjoint({'npoigages'})

    def is_seg_param(self) -> bool:
        """Test if parameter is dimensioned by nsegment.

        :returns: true if parameter is dimensioned by nsegment"""
        return not set(self.__dimensions.keys()).isdisjoint({'nsegment'})

    def outliers(self) -> NamedTuple:
        """Returns the number of values less than or greater than the valid range

        :returns: NamedTuple containing count of values less than and values greater than valid range
        """
        Outliers = namedtuple('Outliers', ['name', 'under', 'over'])

        values_under = 0
        values_over = 0

        if self.__data is not None:
            if self.__minimum is not None:
                values_under = np.count_nonzero(self.__data < self.__minimum)
                # values_under = len(self.__data[self.__data < self.__minimum])

            if self.__maximum is not None:
                values_over = np.count_nonzero(self.__data > self.__maximum)
                # values_over = len(self.__data[self.__data > self.__maximum])

        return Outliers(self.__name, values_under, values_over)

    def remove_by_index(self, dim_name: str, indices: List[int]):
        """Remove columns (nhru or nsegment) from data array given a list of indices.

        :param dim_name: Name of dimension to reduce
        :param indices: List of indices to remove"""

        if isinstance(indices, type(OrderedDict().values())):
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

    def reshape(self, new_dims: OrderedDict):
        """Reshape a parameter, broadcasting existing values as necessary.

        :param new_dims: Dimension names and sizes that will be used to reshape the parameter data
        """

        if self.__data is None:
            # Reshape has no meaning if there is no data to reshape
            return

        if self.dimensions.ndims == 1:
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

        if isinstance(indices, type(OrderedDict().values())):
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
                if self.datatype in [2, 3]:
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
            raise TypeError('Parameter data is not initialized')

    def tostructure(self) -> dict:
        """Returns a dictionary structure of the parameter.

        This is typically used for serializing parameters.

        :returns: dictionary structure of the parameter
        """

        # Return all information about this parameter in the following form
        param = {'name': self.name,
                 'datatype': self.datatype,
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
            if np.array_equal(self.__data[index], value):
                pass
                # print(f'{self.__name}: updated value is equal to the old value')
            else:
                self.__data[index] = value
                self.__modified = True
        else:
            raise TypeError('Parameter data is not initialized')

    def _value_index(self, value: Union[int, float, str]) -> Union[npt.NDArray, None]:
        """Given a scalar value return the indices where there is a match.

        :param value: The value to find in the parameter data array

        :returns: Array of zero-based indices matching the given value
        """

        if self.ndims > 1:
            # TODO: 2021-03-24 PAN - add support for 2D arrays
            print(f'{self.name}: _value_index() does not support 2D arrays yet')
            return None
        else:
            # Returns a list of indices where the data elements match value
            return np.where(self.__data == value)[0]

    # @staticmethod
    # def __str_to_float(data: List[str]) -> List[float]:
    #     """Convert strings to a floats.
    #
    #     :param data: list of data
    #
    #     :returns: array of floats
    #     """
    #
    #     # Convert provide list of data to float
    #     return [float(vv) for vv in data]
    #     # try:
    #     #     return [float(vv) for vv in data]
    #     # except ValueError as ve:
    #     #     print(ve)
    #
    # @staticmethod
    # def __str_to_int(data: List[str]) -> List[int]:
    #     """Converts strings to integers.
    #
    #     :param data: list of data
    #
    #     :returns: array of integers
    #     """
    #
    #     # Convert list of data to integer
    #     try:
    #         return [int(vv) for vv in data]
    #     except ValueError:
    #         # Perhaps it's a float, try converting to float and then integer
    #         tmp = [float(vv) for vv in data]
    #         return [int(vv) for vv in tmp]
    #
    # @staticmethod
    # def __str_to_str(data: List[str]) -> List[str]:
    #     """Null op for string-to-string conversion.
    #
    #     :param data: list of data
    #
    #     :returns: unmodified array of data
    #     """
    #
    #     # nop for list of strings
    #     # 2019-05-22 PAN: For python 3 force string type to byte
    #     #                 otherwise they are treated as unicode
    #     #                 which breaks the write_netcdf() routine.
    #     # 2019-06-26 PAN: Removed the encode because it broke writing the ASCII
    #     #                 parameter files. Instead the conversion to ascii is
    #     #                 handled in the write_netcdf routine of ParameterSet
    #     # data = [dd.encode() for dd in data]
    #     if not all(isinstance(dd, str) for dd in data):
    #         return [str(vv) for vv in data]
    #     return data

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
    #     if not self.ndims:
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
    #     if self.ndims == 2:
    #         data_np = np.array(data_in).reshape((-1, self.dimensions.get_dimsize_by_index(1),), order='F')
    #     elif self.ndims == 1:
    #         data_np = np.array(data_in)
    #     else:
    #         raise ValueError(f'Number of dimensions, {self.ndims}, is not supported')
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
