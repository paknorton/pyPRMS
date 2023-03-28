#!/usr/bin/env python3

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..constants import DATA_TYPES, PTYPE_TO_DTYPE
from ..Exceptions_custom import ControlError

class ControlVariable(object):
    """
    Class object for a single control variable.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, name: str,
                 datatype: int,
                 default: Optional[Union[int, float, str]] = None,
                 description: Optional[str] = None,
                 valid_values: Optional[Dict] = None,
                 value_repr: Optional[str] = None):
        """Initialize a control variable object.

        :param name: Name of control variable
        :param datatype: The datatype of the control variable
        :param default: The default value
        :param description: The description of the control variable
        :param valid_values: The valid values
        :param value_repr: What do the valid values represent (e.g. flag, parameter, etc)
        """

        self.__name = name
        self.__datatype = datatype
        # self.__default = default
        self.default = default
        self.__description = description
        self.__force_default = False
        self.__valid_values = valid_values  # Possible valid values
        self.__value_repr = value_repr  # What do the valid_values represent (e.g. flag, parameter, etc)?

        self.__associated_value = None  # Based on a value what's the associated valid_value?
        self.__values: Union[np.ndarray, None] = None

    def __str__(self) -> str:
        outstr = f'name: {self.name}\n'
        outstr += f'datatype: {self.datatype} ({DATA_TYPES[self.datatype]})\n'

        outstr += f'description: {self.description}\n'
        # if self.default is not None:
        outstr += f'default value: {self.default}\n'

        if self.value_repr:
            outstr += f'values represent: {self.value_repr}\n'

        if self.valid_values:
            outstr += f'valid_values: \n'
            for kk, vv in self.valid_values.items():
                outstr += f'   {kk}: {vv}\n'

        outstr += 'Size of data: '
        if self.values is not None:
            outstr += f'{self.size}\n'
        else:
            outstr += '<empty>\n'

        return outstr

    @property
    def associated_values(self) -> List[str]:
        """Get list of valid values for a control variable.

        :returns: Control variable valid values
        """

        # TODO: this function should be renamed
        assoc_vals = []
        if self.__valid_values is not None:
            # if self.size > 1:
            if isinstance(self.values, np.ndarray):
                for xx in self.values:
                    for vv in self.__valid_values[xx]:
                        assoc_vals.append(vv)
            else:
                for vv in self.__valid_values[str(self.values)]:
                    assoc_vals.append(vv)

        return assoc_vals

    @property
    def datatype(self) -> int:
        """Get the datatype of the control variable.

        :returns: datatype of control variable
        """
        return self.__datatype

    @datatype.setter
    def datatype(self, dtype: int):
        """Sets the datatype of the control variable.

        :param dtype: The datatype for the control variable (1-Integer, 2-Float, 3-Double, 4-String)
        """

        if dtype in DATA_TYPES:
            self.__datatype = dtype
        else:
            print(f'WARNING: Datatype, {dtype}, is not valid.')

    @property
    def default(self) -> Union[int, float, str, np.ndarray, None]:
        """Get default value for control variable.

        :returns: current default value
        """

        if self.__default is not None:
            if self.__default.size > 1:
                return self.__default
            else:
                return self.__default[0]

        return None

    @default.setter
    def default(self, value: Union[int, float, str, None]):
        """Set the default value for the control variable.

        :param value: The default value
        """

        if value is None:
            # Typically value is None when a ControlVariable is first instantiated
            self.__default = value
            return

        # Convert to correct datatype
        if not isinstance(value, (list, np.ndarray)):
            value = [value]

        self.__default = np.array(value, dtype=PTYPE_TO_DTYPE[self.__datatype])

        # Convert datatype first
        # datatype_conv: Dict[int, Callable] = {1: self.__str_to_int, 2: self.__str_to_float,
        #                                       3: self.__str_to_float, 4: self.__str_to_str}
        #
        # if value is None:
        #     # Typically value is None when a ControlVariable is first instantiated
        #     self.__default = value
        #     return
        #
        # if self.__datatype in DATA_TYPES.keys():
        #     value = datatype_conv[self.__datatype](value)
        # else:
        #     err_txt = f'Defined datatype {self.__datatype} for control variable {self.__name} is not valid'
        #     raise TypeError(err_txt)
        #
        # self.__default = np.array(value)

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self,value):
        self.__description = value
    @property
    def force_default(self) -> bool:
        """Get logical value which indicates whether the default value for a
        control variable should always be used instead of the current value.

        """
        return self.__force_default

    @force_default.setter
    def force_default(self, value: Union[bool, int]):
        """Set (or unset) forced use of default value.

        :param value: new force_default value
        """
        self.__force_default = bool(value)

    @property
    def name(self) -> str:
        """Returns the name of the control variable.

        :returns: Name of control variable
        """

        return self.__name

    @property
    def size(self) -> int:
        """Get the number of values for the control variable.

        :returns: Number of values
        """

        if self.__values is not None:
            return self.__values.size
        elif self.__default is not None:
            return self.__default.size
        else:
            # There are no values stored for the control variable
            return 0

    @property
    def valid_values(self) -> Union[Dict, None]:
        """Return the values that are valid for the control variable.

        :returns: Valid values for the control variable
        """

        return self.__valid_values

    @valid_values.setter
    def valid_values(self, data: Dict):
        """Set the values that are valid for the control variable.

        :param data: Valid values for the control variable
        """

        if isinstance(data, dict):
            self.__valid_values = data

    @property
    def value_repr(self) -> Union[str, None]:
        """Get what the control variable value represents.

        A control variable value can represent a flag, interval, or parameter.

        :returns: Control variable representation value
        """

        return self.__value_repr

    @value_repr.setter
    def value_repr(self, data: Union[str, None]):
        """Set the control variable representation.

        A control variable value can represent a flag, interval, or parameter.

        :param data: Representation value
        """
        self.__value_repr = data

    @property
    def values(self) -> Union[np.ndarray, int, float, str, None]:
        """Get the values for the control variable.

        If force_default is True then the default value is returned regardless
        of what the value is set to; otherwise, current value is returned.

        :returns: Value(s) of control variable
        """

        if self.__values is not None:
            if self.__force_default:
                return self.default
            elif self.__values.size == 0:
                raise ControlError(f'Control variable, {self.__name}, has invalid data')
            elif self.__values.size > 1:
                # An array of data (e.g. start_time)
                return self.__values
            else:
                # Single value
                return self.__values[0]
        else:
            return self.default

    @values.setter
    def values(self, data: Union[Sequence[str], str]):
        """Set the value(s) for the control variable.

        :param data: list or string of value(s)
        """

        # TODO: 2021-09-23 PAN In the case of array variables (e.g. start_time),
        #       check that the new array length is the same as the old array length
        # Convert datatype first
        # ptype_to_np = {1: np.int32, 2: np.float32, 3: np.float64, 4: np.chararray}

        # datatype_conv: Dict[int, Callable] = {1: self.__str_to_int, 2: self.__str_to_float,
        #                                       3: self.__str_to_float, 4: self.__str_to_str}

        # if self.__datatype in DATA_TYPES.keys():
        #     data = datatype_conv[self.__datatype](data)
        # else:
        #     raise TypeError(f'Defined datatype {self.__datatype} for parameter {self.__name} is not valid')
        #
        # # Convert to ndarray
        # self.__values = np.array(data)

        if isinstance(data, list):
            self.__values = np.array(data, dtype=PTYPE_TO_DTYPE[self.__datatype])
        elif isinstance(data, np.ndarray):
            if data.dtype == PTYPE_TO_DTYPE[self.__datatype]:
                self.__values = data
            else:
                # Attempt to convert to correct datatype
                self.__values = np.array(data, dtype=PTYPE_TO_DTYPE[self.__datatype])
        else:
            self.__values = np.array([data], dtype=PTYPE_TO_DTYPE[self.__datatype])

