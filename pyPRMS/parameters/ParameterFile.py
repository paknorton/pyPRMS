
# from typing import Any,  Union, Dict, List, OrderedDict as OrderedDictType, Set
from typing import List, Optional, Set

from ..Exceptions_custom import ParameterError
from .ParameterSet import ParameterSet
from ..constants import DIMENSIONS_HDR, PARAMETERS_HDR, VAR_DELIM


class ParameterFile(ParameterSet):

    """Class to handle reading PRMS parameter file format."""

    def __init__(self, filename: str,
                 verbose: Optional[bool] = False,
                 verify: Optional[bool] = True):
        """Create the ParameterFile object.

        :param filename: name of parameter file
        :param verbose: output debugging information
        :param verify: whether to load the master parameters (default=True)
        """

        super(ParameterFile, self).__init__(verbose=verbose, verify=verify)

        # self.__filename = None
        # self.__header = None

        self.__isloaded = False
        self.__updated_parameters: Set[str] = set()
        self.__verbose = verbose
        self.filename = filename

    @property
    def filename(self) -> str:
        """Get parameter filename.

        :returns: name of parameter file
        """

        return self.__filename

    @filename.setter
    def filename(self, name: str):
        """Set the name of the parameter file.

        :param name: name of parameter file
        """

        self.__isloaded = False
        self.__filename = name
        self.__header: List[str] = []  # Initialize the list of file headers

        self._read()

    @property
    def headers(self) -> List[str]:
        """Get the headers from the parameter file.

        :returns: list of headers from parameter file
        """

        return self.__header

    @property
    def updated_parameters(self) -> Set[str]:
        """Get list of parameters that had more than one entry in the parameter file.

        :returns: list of parameters
        """

        return self.__updated_parameters

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
        bounded_parameters = []

        for line in it:
            if line == VAR_DELIM:
                continue
            varname = line.split(' ')[0]

            # Add the parameter
            try:
                if self.master_parameters is not None:
                    self.parameters.add(varname, info=self.master_parameters[varname])

                    if self.master_parameters[varname].minimum == 'bounded':
                        # TODO: The min and max of bounded parameter values will be adjusted later
                        bounded_parameters.append(varname)
                else:
                    self.parameters.add(varname)
            except ParameterError:
                if self.__verbose:
                    print(f'Parameter, {varname}, updated with new values')
                self.__updated_parameters.add(varname)
            except ValueError:
                if self.__verbose:
                    print(f'Parameter, {varname}, is not a valid parameter; skipping.')

                # Skip to the next parameter
                try:
                    while next(it) != VAR_DELIM:
                        pass
                except StopIteration:
                    # Hit end of file
                    pass
                continue

            # Read the dimension names
            ndims = int(next(it))  # number of dimensions for this variable
            dim_names = [next(it) for _ in range(ndims)]

            # Total dimension size declared for parameter in file; it should equal the size of
            # the declared global dimensions.
            dim_size = int(next(it))

            self.parameters.get(varname).datatype = int(next(it))

            # Add the dimensions to the parameter, dimension size is looked up from the global Dimensions object
            for dd in dim_names:
                self.parameters.get(varname).dimensions.add(dd, self.dimensions.get(dd).size)

            # if numval != dim_size:
            if dim_size != self.parameters.get(varname).size:
                # The declared total size doesn't match the total size of the declared dimensions
                print(f'{varname}: Declared total size for parameter does not match the total size of the ' +
                      f'declared dimension(s) ({dim_size} != {self.parameters.get(varname).size}); skipping')

                # Still have to read all the values to skip this properly
                try:
                    while True:
                        cval = next(it)

                        if cval == VAR_DELIM or cval.strip() == '':
                            break
                except StopIteration:
                    # Hit the end of the file
                    pass
                self.parameters.remove(varname)
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

                if len(vals) != dim_size:
                    print(f'{varname}: number of values does not match declared dimension size ' +
                          f'({len(vals)} != {dim_size}); skipping')

                    # Remove the parameter from the dictionary
                    self.parameters.remove(varname)
                else:
                    # Convert the values to the correct datatype
                    # Ignore the type until https://github.com/python/mypy/issues/3004 is fixed
                    self.parameters.get(varname).data = vals    # type: ignore

        for pp in bounded_parameters:
            self._adjust_bounded(pp)

        self.__isloaded = True
