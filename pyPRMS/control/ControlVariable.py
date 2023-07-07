#!/usr/bin/env python3

import datetime
import numpy as np
import re
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..constants import DATA_TYPES, NEW_PTYPE_TO_DTYPE
from pyPRMS.prms_helpers import cond_check, set_date
from ..Exceptions_custom import ControlError

class ControlVariable(object):
    """
    Class object for a single control variable.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18

    def __init__(self, name: str,
                 value = None,
                 meta: Optional[Dict] = None,
                 strict: Optional[bool] = True):
        """Initialize a control variable object.

        :param name: Name of control variable
        :param meta: Metadata of the control variable
        """

        self.__name = name
        self.__values: Union[np.ndarray, None] = None

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
                # The meta must be supplied as an adhoc dictionary
                self.meta = meta

        self.values = value

    def __str__(self) -> str:
        outstr = f'----- ControlVariable -----\n'
        outstr += f'name: {self.name}\n'

        for kk, vv in self.meta.items():
            outstr += f'{kk}: {vv}\n'

        return outstr

        # outstr = f'name: {self.name}\n'
        # if self.meta is not None:
        #     outstr += f'datatype: {self.meta.get("datatype")}\n'
        #     # outstr += f'datatype: {self.datatype} ({DATA_TYPES[self.datatype]})\n'
        #
        #     outstr += f'description: {self.meta["description"]}\n'
        #     outstr += f'default: {self.meta["default"]}\n'
        #
        #     if 'valid_value_type' in self.meta:
        #         outstr += f'valid values represent: {self.meta["valid_value_type"]}\n'
        #
        #     if 'valid_values' in self.meta:
        #         outstr += f'valid_values: \n'
        #         for kk, vv in self.meta['valid_values'].items():
        #             outstr += f'   {kk}: {vv}\n'
        #
        # return outstr

    # @property
    # def associated_values(self) -> List[str]:
    #     """Get list of valid values for a control variable.
    #
    #     :returns: Control variable valid values
    #     """
    #
    #     # TODO: this function should be renamed
    #     assoc_vals = []
    #     # if self.__valid_values is not None:
    #     if 'valid_values' in self.meta:
    #         return list(self.meta['valid_values'][self.values])
    #         # for xx in self.values:
    #         #     for vv in self.__valid_values[xx]:
    #         #         assoc_vals.append(vv)
    #         # else:
    #         #     for vv in self.__valid_values[str(self.values)]:
    #         #         assoc_vals.append(vv)
    #
    #     return assoc_vals

    @property
    def name(self) -> str:
        """Returns the name of the control variable.

        :returns: Name of control variable
        """

        return self.__name

    @property
    def size(self) -> int:
        """Return number of values"""
        if self.__values is None:
            return 0
        elif isinstance(self.__values, np.ndarray):
            return self.__values.size
        else:
            # int, float, str scalars
            return 1

    @property
    def values(self) -> Union[np.ndarray, int, float, str, None]:
        """Get the values for the control variable.

        If force_default is True then the default value is returned regardless
        of what the value is set to; otherwise, current value is returned.

        :returns: Value(s) of control variable
        """

        force_default = self.meta.get('force_default', False)

        if force_default or self.__values is None:
            return self.meta.get('default', None)
        else:
            return self.__values

    @values.setter
    def values(self, data: Union[Sequence[str], str, int, float, datetime.datetime]):
        """Set the value(s) for the control variable.

        :param data: list or string of value(s)
        """

        # if self.meta is not None:
        if data is None:
            self.__values = self.meta.get('default', None)
        else:
            cdtype = NEW_PTYPE_TO_DTYPE[self.meta['datatype']]

            if self.meta['context'] == 'scalar':
                if isinstance(data, list):
                    # Only first value is used for scalars
                    self.__values = np.array(data, dtype=cdtype)[0]
                elif isinstance(data, np.ndarray):
                    if self.meta['datatype'] == 'datetime':
                        self.__values = cdtype(set_date(data))
                    elif data.dtype == cdtype:
                        self.__values = data[0]
                    else:
                        raise TypeError(f'{self.name}: {cdtype} expected but got {data[0].dtype}')
                        # print(f'WARNING: {self.name} - datatype inconsistency; no values added')
                elif ((cdtype == np.str_ and not isinstance(data, str | cdtype)) or
                      (cdtype == np.int32 and not isinstance(data, int | cdtype)) or
                      (cdtype == np.float32 and not isinstance(data, float | cdtype))):
                    raise TypeError(f'{self.name}: {cdtype} expected but got {type(data)}')
                else:
                    self.__values = cdtype(data)
            else:
                if isinstance(data, list):
                    self.__values = np.array(data, dtype=cdtype)
                elif isinstance(data, np.ndarray):
                    if data.dtype == cdtype:
                        # TODO: 20230707 PAN - currently never hit with string arrays
                        self.__values = data
                    else:
                        # Attempt to convert to correct datatype
                        self.__values = np.array(data, dtype=cdtype)
                else:
                    # TODO: 20230707 PAN - currently never hit
                    self.__values = np.array([data], dtype=cdtype)

    @property
    def value_meaning(self) -> Union[str, None]:
        """Returns the meaning for a given value if it exists.

        :returns: Control variable value meaning
        """

        # This will fail for keys that are conditionals (e.g. ">0")
        meaning = self.meta.get('valid_values', None)

        if meaning is None:
            # valid_values key does not exist
            return meaning

        return self._value_meaning_test(self.values, meaning)
        # return meaning.get(self.values, meaning.get(str(self.values), None))

    def _value_meaning_test(self, key, src_dict):
        try:
            return src_dict[key]
        except KeyError:
            # Maybe the key is a string
            try:
                return src_dict[str(key)]
            except KeyError:
                # Maybe one of the keys is a conditional?
                patterns = ['[><]']
                regex = [re.compile('^' + pat).match for pat in patterns]

                tt = {kk: vv for kk, vv in src_dict.items()
                      if any (reg(kk) for reg in regex)}

                if len(tt) > 0:
                    # So there is a conditional
                    for mm in tt:
                        # print(mm.split())
                        if cond_check[mm[0]](key, int(mm[1:])):
                            return src_dict[mm]
                            # print(f'{mm}: {src_dict[mm]}')
        raise ValueError('Invalid control value')


        #     try:
        #         if 'valid_values' in self.meta:
        #             # We want a KeyError here if the key is missing
        #             return self.meta['valid_values'][self.values]
        #
        #         return None
        #     except KeyError:
        #         # Try again but return None if the key is still missing
        #         return self.meta['valid_values'].get(str(self.values), None)
        # return None