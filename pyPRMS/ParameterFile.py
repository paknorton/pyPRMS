
from __future__ import (absolute_import, division, print_function)

from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.ParameterSet import ParameterSet
from pyPRMS.constants import DIMENSIONS_HDR, PARAMETERS_HDR, VAR_DELIM

import functools


class ParameterFile(ParameterSet):

    """Class to handle reading PRMS parameter file format."""

    def __init__(self, filename, verbose=False, verify=True):
        """Create the ParameterFile object.

        :param str filename: name of parameter file
        :param bool verbose: output debugging information
        :param bool verify: whether to load the master parameters (default=True)
        """

        super(ParameterFile, self).__init__(verbose=verbose, verify=verify)

        self.__filename = None
        self.__header = None

        self.__isloaded = False
        self.__updated_params = set()
        self.__verbose = verbose
        self.filename = filename

    @property
    def filename(self):
        """Get parameter filename.

        :returns: name of parameter file
        :rtype: str
        """

        return self.__filename

    @filename.setter
    def filename(self, name):
        """Set the name of the parameter file.

        :param str name: name of parameter file
        """

        self.__isloaded = False
        self.__filename = name
        self.__header = []  # Initialize the list of file headers

        self._read()

    @property
    def headers(self):
        """Get the headers from the parameter file.

        :returns: list of headers from parameter file
        :rtype: list[str]
        """

        return self.__header

    @property
    def updated_params(self):
        """Get list of parameters that had more than one entry in the parameter file.

        :returns: list of parameters
        :rtype: list[str]
        """

        return self.__updated_params

    def _read(self):
        """Read parameter file.
        """

        if self.__verbose:
            print('INFO: Reading parameter file')

        # Read the parameter file into memory and parse it
        infile = open(self.filename, 'r', encoding='ascii')
        rawdata = infile.read().splitlines()
        infile.close()

        it = iter(rawdata)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Grab the header stuff first
        for line in it:
            if line.strip('* ') == DIMENSIONS_HDR:
                break
            self.__header.append(line)

        if self.__verbose:
            print('INFO: headers:')
            print(self.__header)

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
                    print('Parameter, {}, updated with new values'.format(varname))
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
            # If a dimension name does not exist in the list of global dimensions
            # an error occurs.
            arr_shp = [self.dimensions.get(dd).size for dd in dim_tmp]

            # Compute the total size of the parameter
            dim_size = functools.reduce(lambda x, y: x * y, arr_shp)

            # Total dimension size declared for parameter in file; it should be total size of declared dimensions.
            numval = int(next(it))

            self.parameters.get(varname).datatype = int(next(it))

            if self.master_parameters is not None:
                try:
                    master_info = self.master_parameters.parameters[varname]

                    self.parameters.get(varname).units = master_info.units
                    self.parameters.get(varname).description = master_info.description
                    self.parameters.get(varname).help = master_info.help
                    self.parameters.get(varname).modules = master_info.modules
                except KeyError:
                    print('WARNING: {} has no master information'.format(varname))

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
                vals = []
                try:
                    # Read in the data values
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
