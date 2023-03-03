
import xarray as xr
from typing import Optional

from .ParameterSet import ParameterSet


NCF_TO_NHM_TYPES = {'int32': 1, 'float32': 2, 'float64': 3, '|S1': 4}


class ParameterNetCDF(ParameterSet):
    """Read parameter database stored in netCDF format"""

    def __init__(self, filename: str,
                 verbose: Optional[bool] = False,
                 verify: Optional[bool] = True):
        """Initialize ParamDb object.

        :param filename: Path the ParamDb netcdf file
        :param verbose: Output additional debugging information
        :param verify: Verify parameters against master list
        """

        super(ParameterNetCDF, self).__init__(verbose=verbose, verify=verify)
        self.__filename = filename
        self.__verbose = verbose

        # Read the parameters from the parameter database
        self._read()

    def _read(self):
        """Read a paramDb file.
        """

        xr_df = xr.open_dataset(self.__filename, mask_and_scale=False, decode_timedelta=False)

        # Populate the dimensions first
        self.dimensions.add(name='one', size=1)

        for dn, ds in xr_df.dims.items():
            self.dimensions.add(name=str(dn), size=ds)

        # Now add the parameters
        for var in xr_df.variables.keys():
            if self.__verbose:
                print(str(var))

            cparam = xr_df[var].T

            if self.master_parameters is not None:
                self.parameters.add(name=str(var), info=self.master_parameters[var])
            else:
                self.parameters.add(name=str(var),
                                    datatype=NCF_TO_NHM_TYPES[cparam.encoding['dtype']],
                                    units=cparam.attrs['units'],
                                    description=cparam.attrs['description'],
                                    minimum=cparam.attrs['valid_min'],
                                    maximum=cparam.attrs['valid_max'])

            # Add the dimensions for the parameter
            if len(cparam.dims) == 0:
                # Scalar
                self.parameters.get(str(var)).dimensions.add(name='one', size=1)
            else:
                for dim in cparam.dims:
                    self.parameters.get(str(var)).dimensions.add(name=dim, size=self.dimensions.get(dim).size)

            # Add the data
            self.parameters.get(str(var)).data = cparam.values
