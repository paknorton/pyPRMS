import pytest
# import numpy as np
import os
from distutils import dir_util
from pyPRMS import ControlFile
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


class TestControlFile:

    def test_read_control(self, datadir):
        control_file = datadir.join('control.default')

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)

        assert ctl.header == ['$Id:$']

        expected_str = "----- ControlVariable -----\nname: strmflow_module\ndatatype: string\ndescription: Module name for streamflow routing simulation method\ncontext: scalar\ndefault: strmflow\nvalid_value_type: module\nvalid_values: {'muskingum': 'Computes flow in the stream network using the Muskingum routing method (Linsley and others, 1982).', 'muskingum_lake': 'Computes flow in the stream network using the Muskingum routing method and flow and storage in on-channel lake using several methods.', 'muskingum_mann': 'Computes flow in the stream network using the Muskingum routing method with Manning’s N equation.', 'strmflow': 'Computes daily streamflow as the sum of surface runoff, shallow-subsurface flow (interflow), detention reservoir flow, and groundwater flow.', 'strmflow_in_out': 'Routes water between segments in the stream network by setting the outflow to the inflow.'}\n"
        assert ctl.get('strmflow_module').__str__() == expected_str

        expected_arr = ['basin_potet', 'basin_horad', 'basin_orad',
                        'basin_swrad', 'basin_temp', 'basin_tmax',
                        'basin_tmin', 'basin_ppt', 'basin_obs_ppt',
                        'basin_rain', 'basin_snow', 'basin_soil_moist']
        assert ctl.get('basinOutVar_names').values.tolist() == expected_arr

    def test_bad_var_in_file(self, datadir):
        """Bad control variables should be skipped with a warning"""
        control_file = datadir.join('control.bad_var')

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)

        assert not ctl.exists('random_ctl_var')

    def test_bad_num_vals_in_file(self, datadir):
        """Too many values for a variable  should raise ControlError"""
        control_file = datadir.join('control.bad_num_vals')

        prms_meta = MetaData(verbose=True).metadata

        with pytest.raises(ControlError):
            ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)

    def test_dup_var_in_file(self, datadir):
        """Duplicate control variables should be updated with new value and print a warning"""
        control_file = datadir.join('control.dup_var')

        prms_meta = MetaData(verbose=True).metadata

        # If verbose=False warnings about duplicates are not shown
        ctl = ControlFile(control_file, metadata=prms_meta, verbose=True, version=5)

        assert ctl.get('print_debug').values == 4
