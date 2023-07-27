import pytest
import numpy as np
import os
from distutils import dir_util
from pyPRMS import ControlFile
from pyPRMS import ParameterNetCDF
from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.metadata.metadata import MetaData

@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    # 2023-07-18
    # https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture()
def pdb_instance(datadir):
    parameter_file = datadir.join('myparam.nc')

    prms_meta = MetaData(verbose=True).metadata

    pdb = ParameterNetCDF(parameter_file, metadata=prms_meta)
    return pdb


class TestParameterNetCDF:

    def test_read_parameter_netcdf_file(self, datadir):
        # control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.nc')

        prms_meta = MetaData(verbose=True).metadata

        # ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterNetCDF(parameter_file, metadata=prms_meta)

        assert np.isclose(pdb.get('albset_rna').data, 0.8)
