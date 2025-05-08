import pytest
import numpy as np
from pyPRMS import Control
from pyPRMS import ControlFile
from pyPRMS.Exceptions_custom import ControlError
from pyPRMS.metadata.metadata import MetaData


class TestControlFile:

    def test_read_control(self, datadir):
        control_file = datadir / 'control.default'

        prms_meta = MetaData(verbose=False).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)

        assert ctl.header == ['$Id:$']

        expected_str = "----- ControlVariable -----\nname: strmflow_module\ndatatype: string\ndescription: Module name for streamflow routing simulation method\ncontext: scalar\ndefault: strmflow\nvalid_value_type: module\nvalid_values: {'muskingum': 'Computes flow in the stream network using the Muskingum routing method (Linsley and others, 1982).', 'muskingum_lake': 'Computes flow in the stream network using the Muskingum routing method and flow and storage in on-channel lake using several methods.', 'muskingum_mann': 'Computes flow in the stream network using the Muskingum routing method with Manning’s N equation.', 'strmflow': 'Computes daily streamflow as the sum of surface runoff, shallow-subsurface flow (interflow), detention reservoir flow, and groundwater flow.', 'strmflow_in_out': 'Routes water between segments in the stream network by setting the outflow to the inflow.'}\n"
        assert ctl.get('strmflow_module').__str__() == expected_str

        expected_arr = ['basin_potet', 'basin_horad', 'basin_orad',
                        'basin_swrad', 'basin_temp', 'basin_tmax',
                        'basin_tmin', 'basin_ppt', 'basin_obs_ppt',
                        'basin_rain', 'basin_snow', 'basin_soil_moist']
        assert ctl.get('basinOutVar_names').values.tolist() == expected_arr

    def test_diff_control(self, datadir):
        control_file = datadir / 'control.default'

        prms_meta = MetaData(verbose=False).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)

        # Create a instance of a base control class
        ctl_base = Control(metadata=prms_meta)
        expected_diff = {'self_not_other': set(),
                         'other_not_self': set(),
                         'diffs': {
                             'basinOutBaseFileName': {'self': 'output/basin_out_', 'other': 'basinout_path'},
                             'basinOutON_OFF': {'self': 1, 'other': 0},
                             'basinOutVar_names': {
                                 'self': np.array(['basin_potet', 'basin_horad', 'basin_orad', 'basin_swrad',
                                                   'basin_temp', 'basin_tmax', 'basin_tmin', 'basin_ppt',
                                                   'basin_obs_ppt', 'basin_rain', 'basin_snow', 'basin_soil_moist'],
                                                  dtype='<U16'),
                                 'other': 'none'
                             },
                             'basinOutVars': {'self': 12, 'other': 0},
                             'dprst_flag': {'self': 1, 'other': 0},
                             'end_time': {
                                 'self': np.datetime64('2016-09-30T00:00:00.000000'),
                                 'other': np.datetime64('1980-12-31T00:00:00.000000')
                             },
                             'executable_desc': {'self': 'PRMS 5', 'other': 'MOWS'},
                             'executable_model': {'self': './prms', 'other': 'prmsIV'},
                             'nhruOutON_OFF': {'self': 1, 'other': 0},
                             'nsegmentOutON_OFF': {'self': 1, 'other': 0},
                             'param_file': {'self': 'myparam.param', 'other': 'prms.params'},
                             'precip_module': {'self': 'climate_hru', 'other': 'precip_1sta'},
                             'start_time': {
                                 'self': np.datetime64('1980-10-01T00:00:00.000000'),
                                 'other': np.datetime64('1980-01-01T00:00:00.000000')
                             },
                             'strmflow_module': {'self': 'muskingum_mann', 'other': 'strmflow'},
                             'subbasin_flag': {'self': 0, 'other': 1},
                             'temp_module': {'self': 'climate_hru', 'other': 'temp_1sta'}
                         }
                         }

        np.testing.assert_equal(ctl.diff(ctl_base), expected_diff)

    def test_bad_var_in_file(self, datadir):
        """Bad control variables should be skipped with a warning"""
        control_file = datadir / 'control.bad_var'

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)

        assert not ctl.exists('random_ctl_var')

    def test_bad_num_vals_in_file(self, datadir):
        """Too many values for a variable  should raise ControlError"""
        control_file = datadir / 'control.bad_num_vals'

        prms_meta = MetaData(verbose=True).metadata

        with pytest.raises(ControlError):
            ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)

    def test_dup_var_in_file(self, datadir):
        """Duplicate control variables should be updated with new value and print a warning"""
        control_file = datadir / 'control.dup_var'

        prms_meta = MetaData(verbose=True).metadata

        # If verbose=False warnings about duplicates are not shown
        ctl = ControlFile(control_file, metadata=prms_meta, verbose=True)

        assert ctl.get('print_debug').values == 4

    def test_write_control_file(self, datadir, tmp_path):
        """Test if a control file can be written and read back in with the same values"""
        control_file = datadir / 'control.default'

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)
        ctl_vars_orig = ctl.control_variables.keys()

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'control.new'

        ctl.write(out_file)

        prms_meta_chk = MetaData(verbose=True).metadata
        ctl_chk = ControlFile(out_file, metadata=prms_meta_chk, verbose=True)
        ctl_vars_chk = ctl_chk.control_variables.keys()

        # Do both control files have the same variables?
        assert len(set(ctl_vars_orig).symmetric_difference(set(ctl_vars_chk))) == 0

        # Are the values the same for all the variables?
        for cvar in ctl_vars_orig:
            if isinstance(ctl.get(cvar).values, np.ndarray):
                assert (ctl.get(cvar).values == ctl_chk.get(cvar).values).all()
            else:
                assert ctl.get(cvar).values == ctl_chk.get(cvar).values



