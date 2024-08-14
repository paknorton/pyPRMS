#!/usr/bin/env python3

import numpy as np

from pathlib import Path
from typing import Optional, Union
from ..constants import DATA_TYPES, PTYPE_TO_DTYPE, VAR_DELIM
from ..prms_helpers import get_file_iter
from .Control import Control
from ..Exceptions_custom import ControlError


class ControlFile(Control):
    """
    Class which handles the processing of PRMS control files.
    """

    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2019-04-18
    # Description: Class object to handle reading and writing PRMS control files.

    def __init__(self, filename: Union[str, Path], metadata, verbose: Optional[bool] = False,
                 version:Optional[Union[str, int]] = 5):
        super(ControlFile, self).__init__(metadata=metadata, verbose=verbose, version=version)

        self.__verbose = verbose
        self.__isloaded = False
        self.filename = filename

    @property
    def filename(self) -> Union[str, Path]:
        """Get control filename.

        :returns: Name of control file
        """
        return self.__filename

    @filename.setter
    def filename(self, filename: Union[str, Path]):
        """Set the name of the control file and read it in.

        :param filename: Name of the control file
        """

        self.__isloaded = False
        self.__filename = filename
        self._read()

    def _read(self):
        """Read contents of control file.

        Reads the contents of a control file into the class.
        """

        if self.__verbose:
            chk_vars = []

        self.__isloaded = False

        it = get_file_iter(self.filename)
        header_tmp = []

        for fidx, line in enumerate(it):
            if fidx == 0:
                header_tmp.append(line)
                continue
            elif line == VAR_DELIM:
                continue
            else:
                # We're dealing with a control parameter/variable
                # We're in a parameter section
                varname = line.split(' ')[0]

                if self.__verbose:
                    if varname in chk_vars:
                        print(f'WARNING: {varname} already exists')
                    chk_vars.append(varname)

                numval = int(next(it))  # number of values for this variable
                valuetype = int(next(it))  # Variable type (1 - integer, 2 - float, 4 - character)

                try:
                    # if self.get(varname).context == 'scalar' and varname not in ['start_time', 'end_time']:
                    if self.get(varname).meta['context'] == 'scalar':
                        if numval == 1:  # varname not in ['start_time', 'end_time']:
                            vals = PTYPE_TO_DTYPE[valuetype](next(it))
                        else:
                            # Currently only start_time and end_time are scalars with numval > 1
                            vals = np.zeros(numval, dtype=PTYPE_TO_DTYPE[valuetype])
                            for idx in range(0, numval):
                                # NOTE: string-float to int works but float to int does not
                                vals[idx] = next(it)
                    elif self.get(varname).meta['context'] == 'array':
                        if valuetype == 4:
                            # Arrays of strings should be objects
                            vals = np.zeros(numval, dtype=object)
                        else:
                            vals = np.zeros(numval, dtype=PTYPE_TO_DTYPE[valuetype])

                        for idx in range(0, numval):
                            # NOTE: string-float to int works but float to int does not
                            vals[idx] = next(it)
                    else:
                        print(f'WARNING: {varname} has context={self.get(varname).meta["context"]} which is not supported')

                    # After reading expected values make sure there aren't more values
                    # before the next delimiter.
                    try:
                        cnt = numval
                        while next(it) != VAR_DELIM:
                            cnt += 1

                        if cnt > numval:
                            raise ControlError(f'{varname}: too many values specified')
                            # print(f'WARNING: Too many values specified for {varname}')
                            # print(f'      {numval} expected, {cnt} given')
                            # print(f'       Keeping first {numval} values')
                    except StopIteration:
                        # Hit the end of the file
                        pass

                    self.get(varname).values = vals

                except ValueError as err:
                    print(f'WARNING: {varname} is not a valid control variable')
                    print(err)
                    while next(it) != VAR_DELIM:
                        pass

        self.header = header_tmp
        self.__isloaded = True

    # def clear_parameter_group(self, group_name):
    #     """Clear a parameter group.
    #
    #     Given a single parameter group name will clear out values for that parameter
    #        and all related parameters.
    #
    #     Args:
    #         group_name: The name of a group of related parameters.
    #             One of 'statVar', 'aniOut', 'mapOut', 'dispVar', 'nhruOut'.
    #     """
    #
    #     groups = {'aniOut': {'naniOutVars': 0, 'aniOutON_OFF': 0, 'aniOutVar_names': []},
    #               'dispVar': {'ndispGraphs': 0, 'dispVar_element': [], 'dispVar_names': [], 'dispVar_plot': []},
    #               'mapOut': {'nmapOutVars': 0, 'mapOutON_OFF': 0, 'mapOutVar_names': []},
    #               'nhruOut': {'nhruOutVars': 0, 'nhruOutON_OFF': 0, 'nhruOutVar_names': []},
    #               'statVar': {'nstatVars': 0, 'statsON_OFF': 0, 'statVar_element': [], 'statVar_names': []}}
    #
    #     for (kk, vv) in iteritems(groups[group_name]):
    #         if kk in self.__controldict:
    #             self.replace_values(kk, vv)
