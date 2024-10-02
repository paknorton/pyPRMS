import pytest
import numpy as np
from pyPRMS import ParameterNetCDF
from pyPRMS.metadata.metadata import MetaData


@pytest.fixture()
def pdb_instance(datadir):
    parameter_file = datadir / 'myparam.nc'

    prms_meta = MetaData(verbose=True).metadata

    pdb = ParameterNetCDF(parameter_file, metadata=prms_meta)
    return pdb


class TestParameterNetCDF:

    def test_read_parameter_netcdf_file(self, pdb_instance):
        """Check basic reading of netcdf parameter file"""
        assert np.isclose(pdb_instance.get('albset_rna').data, 0.8)
