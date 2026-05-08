import pytest
import numpy as np
from pyPRMS import Control
from pyPRMS import MetaData
from pyPRMS.Exceptions_custom import ControlError


@pytest.fixture(scope='class')
def control_object():
    prms_meta = MetaData(verbose=False).metadata
    ctl = Control(metadata=prms_meta, verbose=True)

    return ctl


@pytest.fixture(scope='class')
def metadata_ctl():
    prms_meta = MetaData(verbose=False).metadata['control']

    return prms_meta


class TestControl:

    def test_control_default_metadata(self, control_object, datadir, tmp_path):
        """Test the default control metadata CSV file"""
        ctl_metadata_orig_file = datadir / 'ctl_metadata_default.csv'

        with open(ctl_metadata_orig_file, 'r') as f:
            lines_orig = f.readlines()

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'ctl_metadata_default_test.csv'

        control_object.write_metadata_csv(out_file)

        with open(out_file, 'r') as f:
            lines_chk = f.readlines()

        assert lines_orig == lines_chk, 'Control metadata CSV file does not match expected'

    def test_getitem(self, control_object):
        assert control_object['print_debug'].values == 0

    def test_to_dict(self, control_object):
        expected = {'aniOutON_OFF': 0,
                    'ani_output_file': 'animation.out',
                    'naniOutVars': 0,
                    'aniOutVar_names': 'none',
                    'basinOutON_OFF': 0,
                    'basinOutBaseFileName': 'basinout_path',
                    'basinOutVars': 0,
                    'basinOutVar_names': 'none',
                    'basinOut_freq': 1,
                    'nhruOutNcol': 0,
                    'nhruOutON_OFF': 0,
                    'nhruOutBaseFileName': 'nhru_summary_',
                    'nhruOutVars': 0,
                    'nhruOut_format': 1,
                    'nhruOut_freq': 1,
                    'outputSelectDatesON_OFF': 0,
                    'nhruOutVar_names': 'none',
                    'selectDatesFileName': 'selectDates.in',
                    'nsegmentOutON_OFF': 0,
                    'nsegmentOutBaseFileName': 'nsegment_summary_',
                    'nsegmentOutVars': 0,
                    'nsegmentOutVar_names': 'none',
                    'nsegmentOut_format': 1,
                    'nsegmentOut_freq': 1,
                    'nsubOutON_OFF': 0,
                    'nsubOutBaseFileName': 'nsub_summary_',
                    'nsubOutVars': 0,
                    'nsubOutVar_names': 'none',
                    'nsubOut_format': 1,
                    'nsubOut_freq': 1,
                    'statsON_OFF': 0,
                    'stat_var_file': 'statvar.out',
                    'nstatVars': 0,
                    'statVar_element': 'none',
                    'statVar_names': 'none',
                    'cascade_flag': 0,
                    'cascadegw_flag': 0,
                    'cbh_check_flag': 0,
                    'csvON_OFF': 0,
                    'dispGraphsBuffSize': 50,
                    'ndispGraphs': 0,
                    'dispVar_element': 'none',
                    'dispVar_names': 'none',
                    'dispVar_plot': 'none',
                    'mapOutON_OFF': 0,
                    'nmapOutVars': 0,
                    'mapOutVar_names': 'none',
                    'dprst_flag': 0,
                    'dprst_transferON_OFF': 0,
                    'dyn_covden_flag': 0,
                    'dyn_covtype_flag': 0,
                    'dyn_dprst_flag': 0,
                    'dyn_fallfrost_flag': 0,
                    'dyn_imperv_flag': 0,
                    'dyn_intcp_flag': 0,
                    'dyn_potet_flag': 0,
                    'dyn_radtrncf_flag': 0,
                    'dyn_snareathresh_flag': 0,
                    'dyn_soil_flag': 0,
                    'dyn_springfrost_flag': 0,
                    'dyn_sro2dprst_imperv_flag': 0,
                    'dyn_sro2dprst_perv_flag': 0,
                    'dyn_transp_flag': 0,
                    'dyn_transp_on_flag': 0,
                    'dynamic_param_log_file': 'dynamic_parameter.out',
                    'external_transferON_OFF': 0,
                    'gwr_swale_flag': 0,
                    'gwr_transferON_OFF': 0,
                    'init_vars_from_file': 0,
                    'lake_transferON_OFF': 0,
                    'orad_flag': 1,
                    'parameter_check_flag': 0,
                    'print_debug': 0,
                    'prms_warmup': 0,
                    'save_vars_to_file': 0,
                    'segment_transferON_OFF': 0,
                    'stream_temp_shade_flag': 0,
                    'strmtemp_humidity_flag': 0,
                    'subbasin_flag': 1,
                    'end_time': np.datetime64('1980-12-31T00:00:00.000000'),
                    'start_time': np.datetime64('1980-01-01T00:00:00.000000'),
                    'initial_deltat': 24.0,
                    'covden_sum_dynamic': 'dyn_covden_sum',
                    'covden_win_dynamic': 'dyn_covden_win',
                    'covtype_dynamic': 'dyn_cov_type.param',
                    'csv_output_file': 'prms_summary.csv',
                    'dprst_depth_dynamic': 'dyn_dprst_depth.param',
                    'dprst_frac_dynamic': 'dyn_dprst_frac.param',
                    'dprst_transfer_file': 'dprst.transfer',
                    'et_module': 'potet_jh',
                    'executable_desc': 'MOWS',
                    'executable_model': 'prmsIV',
                    'external_transfer_file': 'ext.transfer',
                    'fallfrost_dynamic': 'dyn_fall_frost.param',
                    'gwr_transfer_file': 'gwr.transfer',
                    'humidity_day': 'humidity.day',
                    'imperv_frac_dynamic': 'dyn_hru_percent_imperv.param',
                    'imperv_stor_dynamic': 'dyn_imperv_stor_max.param',
                    'lake_transfer_file': 'lake.transfer',
                    'model_mode': 'PRMS5',
                    'model_output_file': 'prms.out',
                    'param_file': 'prms.params',
                    'potet_day': 'potet.day',
                    'potetcoef_dynamic': 'dyn_potet_coef.param',
                    'precip_day': 'precip.day',
                    'precip_map_file': 'precip.map',
                    'precip_module': 'precip_1sta',
                    'radtrncf_dynamic': 'dyn_rad_trncf.param',
                    'segment_transfer_file': 'seg.transfer',
                    'snareathresh_dynamic': 'dyn_snarea_thresh.param',
                    'snow_intcp_dynamic': 'dyn_snow_intcp.param',
                    'soilmoist_dynamic': 'dyn_soil_moist.param',
                    'soilrechr_dynamic': 'dyn_soil_rechr.param',
                    'soilzone_module': 'soilzone',
                    'solrad_module': 'ddsolrad',
                    'springfrost_dynamic': 'dyn_spring_frost.param',
                    'srain_intcp_dynamic': 'dyn_srain_intcp.param',
                    'sro2dprst_imperv_dynamic': 'dyn_sro_to_dprst_imperv.param',
                    'sro2dprst_perv_dynamic': 'dyn_sro_to_dprst_perv.param',
                    'srunoff_module': 'srunoff_smidx',
                    'strmflow_module': 'strmflow',
                    'swrad_day': 'swrad.day',
                    'temp_module': 'temp_1sta',
                    'tmax_day': 'tmax.day',
                    'tmin_day': 'tmin.day',
                    'tmax_map_file': 'tmax.map',
                    'tmin_map_file': 'tmin.map',
                    'transp_day': 'transp.day',
                    'transp_module': 'transp_tindex',
                    'transp_on_dynamic': 'dyn_transp_on.param',
                    'transpbeg_dynamic': 'dyn_transp_beg.param',
                    'transpend_dynamic': 'dyn_transp_end.param',
                    'var_init_file': 'prms_ic.in',
                    'var_save_file': 'prms_ic.out',
                    'windspeed_day': 'windspeed.day',
                    'wrain_intcp_dynamic': 'dyn_wrain_intcp.param',
                    'data_file': 'sf_data',
                    'snarea_curve_flag': 0,
                    'soilzone_aet_flag': 0,
                    'stream_temp_flag': 0,
                    'albedo_cbh_flag': 0,
                    'albedo_day': 'albedo.day',
                    'cloud_cover_cbh_flag': 0,
                    'cloud_cover_day': 'cloudcover.day',
                    'humidity_cbh_flag': 0,
                    'snow_cloudcover_flag': 0,
                    'windspeed_cbh_flag': 0,
                    'frozen_flag': 0,
                    'glacier_flag': 0,
                    'mbInit_flag': 0}

        assert control_object.to_dict() == expected

    def test_control_read_method_is_abstract(self, control_object):
        """The Control class _read() method is abstract"""
        with pytest.raises(NotImplementedError):
            control_object._read()

    def test_default_header(self, control_object):
        """The default control object should have no header"""

        assert control_object.header is None

    def test_set_header_with_str(self, control_object):
        """Set the header with a string"""
        expected_header = 'Cool header'
        control_object.header = expected_header
        assert control_object.header == [expected_header]

    def test_set_header_with_list(self, control_object):
        """Set header with a list"""
        expected_header = ['Header line one', 'Header line two']
        control_object.header = expected_header
        assert control_object.header == expected_header

    def test_set_header_with_none(self, control_object):
        """Set the header to None"""
        control_object.header = None
        assert control_object.header is None

    def test_cbh_files(self, control_object):
        """Check the default set of CBH files"""
        expected = ['albedo.day',
                    'cloudcover.day',
                    'humidity.day',
                    'potet.day',
                    'precip.day',
                    'swrad.day',
                    'tmax.day',
                    'tmin.day',
                    'transp.day',
                    'windspeed.day']
        assert control_object.cbh_files == expected

    def test_default_modules(self, control_object):
        """Check the default set of modules is correct"""
        expected = {'et_module': 'potet_jh',
                    'precip_module': 'precip_1sta',
                    'soilzone_module': 'soilzone',
                    'solrad_module': 'ddsolrad',
                    'srunoff_module': 'srunoff_smidx',
                    'strmflow_module': 'strmflow',
                    'temp_module': 'temp_1sta',
                    'transp_module': 'transp_tindex',
                    'basin_module': 'basin',
                    'intcp_module': 'intcp',
                    'obs_module': 'obs',
                    'snow_module': 'snowcomp',
                    'gw_module': 'gwflow'}
        assert control_object.modules == expected

    def test_default_additional_modules(self, control_object):
        assert control_object.additional_modules == ['basin_sum', 'subbasin']

    def test_all_additional_modules(self, control_object):
        expected = ['basin_sum',
                    'basin_summary',
                    'map_results',
                    'nhru_summary',
                    'nsegment_summary',
                    'nsub_summary',
                    'stream_temp',
                    'subbasin']

        control_object.get('print_debug').values = 4
        control_object.get('basinOutON_OFF').values = 2
        control_object.get('mapOutON_OFF').values = 1
        control_object.get('nhruOutON_OFF').values = 1
        control_object.get('nsegmentOutON_OFF').values = 1
        control_object.get('nsubOutON_OFF').values = 1
        control_object.get('stream_temp_flag').values = 1

        assert control_object.additional_modules == expected

    def test_get_nonexistent_variable(self, control_object):
        """Non-existent variables should raise ValueError"""

        with pytest.raises(ValueError):
            control_object.get('some_var')

    @pytest.mark.parametrize('name', ['bad_name'])
    def test_add_invalid_variable(self, control_object, metadata_ctl, name):
        """Add an invalid control variable name"""

        with pytest.raises(ValueError):
            control_object.add(name=name)

    def test_add_duplicate_variable(self, control_object, metadata_ctl):
        """Add a duplicate control variable"""

        with pytest.raises(ControlError):
            control_object.add(name='et_module')

    def test_remove_variable(self, control_object):
        control_object.remove('albedo_day')
        assert not control_object.exists('albedo_day')

    def test_no_dynamic_parameters(self, control_object):
        """Default control object should have no dynamic parameters"""

        assert (not control_object.has_dynamic_parameters and
                control_object.dynamic_parameters == [])

    def test_set_dynamic_parameter(self, control_object):
        """Check that dynamic parameters are reported correctly"""

        control_object.get('dyn_transp_flag').values = 3

        assert control_object.has_dynamic_parameters and control_object.dynamic_parameters == ['transp_beg', 'transp_end']