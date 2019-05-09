
from __future__ import (absolute_import, division, print_function)

# import os
# import xml.dom.minidom as minidom
# import xml.etree.ElementTree as xmlET

from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import DIMENSIONS_HDR, PARAMETERS_HDR, VAR_DELIM

import functools


class ParameterFile(ParameterSet):
    def __init__(self, filename, verbose=False):
        super(ParameterFile, self).__init__()

        self.__filename = None
        self.__header = None

        self.__isloaded = False
        self.__updated_params = set()
        self.__verbose = verbose
        self.filename = filename

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, name):
        self.__isloaded = False
        self.__filename = name
        self.__header = []  # Initialize the list of file headers

        self._read()

    @property
    def headers(self):
        """Returns the headers read from the parameter file"""
        return self.__header

    @property
    def updated_params(self):
        return self.__updated_params

    def _read(self):
        # Read the parameter file into memory and parse it
        infile = open(self.filename, 'r')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Grab the header stuff first
        for line in it:
            if line.strip('* ') == DIMENSIONS_HDR:
                break
            self.__header.append(line)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Now process the dimensions
        for line in it:
            if line.strip('* ') == PARAMETERS_HDR:
                break
            if line == VAR_DELIM:
                continue

            # Add dimension - all dimensions are scalars
            self.dimensions.add(line, int(next(it)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Lastly process the parameters
        for line in it:
            if line == VAR_DELIM:
                continue
            varname = line.split(' ')[0]

            try:
                self.parameters.add(varname)
            except ParameterError:
                if self.__verbose:
                    print(f'Parameter, {varname}, updated with new values')
                self.__updated_params.add(varname)
                # print('%s: Duplicate parameter name.. skipping' % varname)

                # Skip to the next parameter
                # try:
                #     while next(it) != VAR_DELIM:
                #         pass
                # except StopIteration:
                #     # Hit end of file
                #     pass
                # continue

            # Read in the dimension names
            ndims = int(next(it))  # number of dimensions for this variable
            dim_tmp = [next(it) for _ in range(ndims)]

            # Lookup dimension size for each dimension name
            arr_shp = [self.dimensions.get(dd).size for dd in dim_tmp]

            # Compute the total size of the parameter
            dim_size = functools.reduce(lambda x, y: x * y, arr_shp)

            # Total dimension size declared for parameter in file; it should be total size of declared dimensions.
            numval = int(next(it))

            self.parameters.get(varname).datatype = int(next(it))

            # Add the dimensions to the parameter, dimension size is looked up from the global Dimensions object
            for dd in dim_tmp:
                self.parameters.get(varname).dimensions.add(dd, self.dimensions.get(dd).size)

            if numval != dim_size:
                # The declared total size doesn't match the total size of the declared dimensions
                print('{}: Declared total size for parameter does not match the total size of the ' +
                      'declared dimension(s) ({} != {}).. skipping'.format(varname, numval, dim_size))

                # Still have to read all the values to skip this properly
                try:
                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                except StopIteration:
                    # Hit the end of the file
                    pass
                self.parameters.del_param(varname)
            else:
                # Check if number of values written match the number of values declared
                try:
                    # Read in the data values
                    vals = []

                    while True:
                        cval = next(it)

                        if cval[0:4] == VAR_DELIM or cval.strip() == '':
                            break
                        vals.append(cval)
                except StopIteration:
                    # Hit the end of the file
                    pass

                if len(vals) != numval:
                    print('{}: number of values does not match dimension size ' +
                          '({} != {}).. skipping'.format(varname, len(vals), numval))

                    # Remove the parameter from the dictionary
                    self.parameters.remove(varname)
                else:
                    # Convert the values to the correct datatype
                    self.parameters.get(varname).data = vals

        self.__isloaded = True
