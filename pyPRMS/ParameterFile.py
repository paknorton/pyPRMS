
from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

# from collections import OrderedDict

from pyPRMS.Exceptions_custom import ParameterError
from pyPRMS.Parameters import Parameters
from pyPRMS.Dimensions import Dimensions
from pyPRMS.constants import DIMENSIONS_HDR, PARAMETERS_HDR, VAR_DELIM


class ParameterFile(object):
    def __init__(self, filename):
        self.__filename = None
        self.__parameters = None
        self.__dimensions = None
        self.__header = None

        self.__isloaded = False
        self.filename = filename

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, fname):
        self.__isloaded = False
        self.__filename = fname
        self.__parameters = Parameters()   # Initialize the parameter dictionary
        self.__dimensions = Dimensions()    # Initialize the dimensions object
        self.__header = []  # Initialize the list of file headers

        self._read()

    @property
    def dimensions(self):
        return self.__dimensions.dimensions

    @property
    def headers(self):
        """Returns the headers read from the parameter file"""
        return self.__header

    @property
    def parameters(self):
        return self.__parameters

    def get_dimsize(self, dimname):
        # Return the size of the specified global dimension
        # TODO: This function may need to be renamed to make it less vague
        return self.__dimensions.__getattr__(dimname).size

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
            self.__dimensions.add_dimension(line, int(next(it)))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Lastly process the parameters
        for line in it:
            if line == VAR_DELIM:
                continue
            varname = line.split(' ')[0]

            try:
                self.__parameters.add_param(varname)
            except ParameterError:
                print('%s: Duplicate parameter name.. skipping' % varname)

                # Skip to the next parameter
                try:
                    while next(it) != VAR_DELIM:
                        pass
                except StopIteration:
                    # Hit end of file
                    pass
                continue

            # Read in the dimension names
            ndims = int(next(it))  # number of dimensions for this variable
            dim_tmp = [next(it) for _ in range(ndims)]

            # Lookup dimension size for each dimension name
            arr_shp = [self.__dimensions.__getattr__(dd).size for dd in dim_tmp]

            # Compute the total size of the parameter
            dim_size = reduce(lambda x, y: x * y, arr_shp)

            # Total dimensin size declared for parameter in file; it should be total size of declared dimensions.
            numval = int(next(it))

            self.__parameters.__getattr__(varname).datatype = int(next(it))
            # self.__parameters[varname].datatype = int(next(it))

            for dd in dim_tmp:
                self.__parameters.__getattr__(varname).add_dimension(dd, self.__dimensions.__getattr__(dd).size)

            if numval != dim_size:
                # The declared total size doesn't match the total size of the declared dimensions
                print('%s: Declared total size for parameter does not match the total size of the declared dimension(s) (%d != %d).. skipping' % (varname, numval, dim_size))

                # Still have to read all the values to skip this properly
                try:
                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                except StopIteration:
                    # Hit the end of the file
                    pass
                self.__parameters.del_param(varname)
            else:
                # Check if number of values written match the number of values declared
                try:
                    # Read in the data values
                    vals = []

                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                        vals.append(cval)
                except StopIteration:
                    # Hit the end of the file
                    pass

                if len(vals) != numval:
                    print('%s: number of values does not match dimension size (%d != %d).. skipping' %
                          (varname, len(vals), numval))

                    # Remove the parameter from the dictionary
                    self.__parameters.del_param(varname)
                else:
                    # Convert the values to the correct datatype
                    self.__parameters.__getattr__(varname).data = vals

        # Build the vardict dictionary (links varname to array index in self.__paramdict)
        # self.rebuild_vardict()
        self.__isloaded = True
