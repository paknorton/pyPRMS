import pytest
import numpy as np
import os
import pandas as pd
from distutils import dir_util
from pyPRMS import ControlFile
from pyPRMS import ParamDb
from pyPRMS import ParameterFile
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
    parameter_file = datadir.join('myparam.param')

    prms_meta = MetaData(verbose=True).metadata

    pdb = ParameterFile(parameter_file, metadata=prms_meta)
    return pdb


class TestParameterFile:

    def test_read_parameter_file(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param')

        prms_meta = MetaData(verbose=True).metadata

        # ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)

        expected_headers = ['Written by Bandit version 0.8.7',
                           'ParamDb revision: https///code.usgs.gov/wma/national-iwaas/nhm/nhm-applications/nhm-v1.1-conus/paramdb_v1.1_gridmet_CONUS/commit/1ffad3a9e33473290efaaa472a90b42a12e28e1f']
        assert pdb.headers == expected_headers

    def test_parameter_file_write(self, datadir, tmp_path):
        parameter_file = datadir.join('myparam.param')

        prms_meta_orig = MetaData(verbose=True).metadata
        pdb_orig = ParameterFile(parameter_file, metadata=prms_meta_orig)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'param_test.param'

        pdb_orig.write_parameter_file(out_file, header=pdb_orig.headers)

        prms_meta_chk = MetaData(version=5, verbose=True).metadata
        pdb_chk = ParameterFile(out_file, metadata=prms_meta_chk)

        # Same headers?
        assert pdb_orig.headers == pdb_chk.headers

        # Do they have the same dimensions/sizes?
        orig_dims = pdb_orig.dimensions
        chk_dims = pdb_chk.dimensions

        assert len(set(orig_dims.keys()).symmetric_difference(set(chk_dims.keys()))) == 0, 'Different number of dimensions'

        for cdim in orig_dims.keys():
            assert orig_dims[cdim].size == chk_dims[cdim].size, f'{cdim}: Dimension sizes different'

        # Do they have the parameters/data?
        orig_params = pdb_orig.parameters
        chk_params = pdb_chk.parameters

        assert len(set(orig_params.keys()).symmetric_difference(set(chk_params.keys()))) == 0, 'Different number of parameters'

        for cparam in orig_params.keys():
            # if isinstance(orig_params[cparam].data, np.ndarray):
            assert (orig_params[cparam].data_raw == chk_params[cparam].data_raw).all(), f'{cparam}: Parameter values different'

    def test_parameter_file_write_header_check(self, datadir, tmp_path):
        parameter_file = datadir.join('myparam.param')

        prms_meta_orig = MetaData(verbose=True).metadata
        pdb_orig = ParameterFile(parameter_file, metadata=prms_meta_orig)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'param_test.param'

        with pytest.raises(ValueError):
            pdb_orig.write_parameter_file(out_file, header=['line 1', 'line 2', 'line 3'])

        pdb_orig.write_parameter_file(out_file, header=['line 1'])
        prms_meta_chk = MetaData(version=5, verbose=True).metadata
        pdb_chk = ParameterFile(out_file, metadata=prms_meta_chk)

        assert pdb_chk.headers == ['Written by pyPRMS', 'line 1'], 'Header not written correctly'

        pdb_orig.write_parameter_file(out_file, header=None)
        prms_meta_chk = MetaData(version=5, verbose=True).metadata
        pdb_chk = ParameterFile(out_file, metadata=prms_meta_chk)

        assert pdb_chk.headers == ['Written by pyPRMS', 'Comment: It is all downhill from here'], 'Default header not written correctly'

    def test_paramdb_write(self, datadir, tmp_path):
        parameter_file = datadir.join('myparam.param')

        prms_meta_orig = MetaData(verbose=True).metadata
        pdb_orig = ParameterFile(parameter_file, metadata=prms_meta_orig)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'paramdb_test'

        pdb_orig.write_paramdb(out_file)

        prms_meta_chk = MetaData(version=5, verbose=True).metadata
        pdb_chk = ParamDb(out_file, metadata=prms_meta_chk)

        # Do they have the same dimensions/sizes?
        orig_dims = pdb_orig.dimensions
        chk_dims = pdb_chk.dimensions

        assert len(set(orig_dims.keys()).symmetric_difference(set(chk_dims.keys()))) == 0, 'Different number of dimensions'

        for cdim in orig_dims.keys():
            assert orig_dims[cdim].size == chk_dims[cdim].size, f'{cdim}: Dimension sizes different'

        # Do they have the parameters/data?
        orig_params = pdb_orig.parameters
        chk_params = pdb_chk.parameters

        assert len(set(orig_params.keys()).symmetric_difference(set(chk_params.keys()))) == 0, 'Different number of parameters'

        for cparam in orig_params.keys():
            # if isinstance(orig_params[cparam].data, np.ndarray):
            assert (orig_params[cparam].data_raw == chk_params[cparam].data_raw).all(), f'{cparam}: Parameter values different'

    def test_parameter_netcdf_write(self, datadir, tmp_path):
        parameter_file = datadir.join('myparam.param')

        prms_meta_orig = MetaData(verbose=True).metadata
        pdb_orig = ParameterFile(parameter_file, metadata=prms_meta_orig)

        out_path = tmp_path / 'run_files'
        out_path.mkdir()
        out_file = out_path / 'parameter_netcdf_test.nc'

        pdb_orig.write_parameter_netcdf(out_file)

        prms_meta_chk = MetaData(version=5, verbose=True).metadata
        pdb_chk = ParameterNetCDF(out_file, metadata=prms_meta_chk)

        # Do they have the same dimensions/sizes?
        orig_dims = pdb_orig.dimensions
        chk_dims = pdb_chk.dimensions

        print(set(orig_dims.keys()).symmetric_difference(set(chk_dims.keys())))
        assert len(set(orig_dims.keys()).symmetric_difference(set(chk_dims.keys()))) == 0, 'Different number of dimensions'

        for cdim in orig_dims.keys():
            assert orig_dims[cdim].size == chk_dims[cdim].size, f'{cdim}: Dimension sizes different'

        # Do they have the parameters/data?
        orig_params = pdb_orig.parameters
        chk_params = pdb_chk.parameters

        assert len(set(orig_params.keys()).symmetric_difference(set(chk_params.keys()))) == 0, 'Different number of parameters'

        for cparam in orig_params.keys():
            # if isinstance(orig_params[cparam].data, np.ndarray):
            assert (orig_params[cparam].data_raw == chk_params[cparam].data_raw).all(), f'{cparam}: Parameter values different'


    def test_parameters_unneeded_parameters(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param')

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)
        pdb.control = ctl

        expected_headers = ['Written by Bandit version 0.8.7',
                           'ParamDb revision: https///code.usgs.gov/wma/national-iwaas/nhm/nhm-applications/nhm-v1.1-conus/paramdb_v1.1_gridmet_CONUS/commit/1ffad3a9e33473290efaaa472a90b42a12e28e1f']
        assert pdb.headers == expected_headers

        assert pdb.unneeded_parameters == {'lat_temp_adj', 'azrh', 'width_alpha', 'alte', 'seg_lat', 'vow',
                                           'albedo', 'seg_elev', 'width_m', 'altw', 'stream_tave_init', 'vce',
                                           'maxiter_sntemp', 'voe', 'vdemn', 'vdwmn', 'vhw', 'seg_humidity',
                                           'gw_tau', 'vcw', 'ss_tau', 'vdemx', 'vhe', 'melt_temp', 'vdwmx'}

    def test_add_missing_parameters(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param')

        prms_meta = MetaData(verbose=True).metadata

        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)
        pdb.control = ctl

        missing = pdb.missing_params
        print(missing)

        intial_params = set(pdb.parameters.keys())

        pdb.add_missing_parameters()

        assert set(pdb.parameters.keys()) == intial_params.union(missing - {'nhm_deplcrv'})

    def test_parameters_remove_unneeded(self, pdb_instance):
        remove_list = pdb_instance.unneeded_parameters
        initial_params = set(pdb_instance.parameters.keys())

        pdb_instance.remove(remove_list)

        assert set(pdb_instance.parameters.keys()) == initial_params - remove_list

    def test_read_parameter_file_dup_entry(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param_dup')

        prms_meta = MetaData(verbose=True).metadata

        # ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)

        expected_headers = ['Written by Bandit version 0.8.7',
                           'ParamDb revision: https///code.usgs.gov/wma/national-iwaas/nhm/nhm-applications/nhm-v1.1-conus/paramdb_v1.1_gridmet_CONUS/commit/1ffad3a9e33473290efaaa472a90b42a12e28e1f']
        assert pdb.headers == expected_headers
        assert (pdb.get('width_alpha').data == np.array([66.78, 69.95, 38.75, 67.77, 69.78, 40.2 , 11.61],
                                                        dtype=np.float32)).all()
        assert pdb.updated_parameters == {'width_alpha'}

    def test_read_parameter_file_invalid_param(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param_invalid')

        prms_meta = MetaData(verbose=True).metadata

        # ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)

        expected_headers = ['Written by Bandit version 0.8.7',
                           'ParamDb revision: https///code.usgs.gov/wma/national-iwaas/nhm/nhm-applications/nhm-v1.1-conus/paramdb_v1.1_gridmet_CONUS/commit/1ffad3a9e33473290efaaa472a90b42a12e28e1f']
        assert pdb.headers == expected_headers
        assert not pdb.exists('width_ft')

    def test_read_parameter_file_too_many_values(self, datadir):
        control_file = datadir.join('control.default.bandit')
        parameter_file = datadir.join('myparam.param_too_many_values')

        prms_meta = MetaData(verbose=True).metadata

        # ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)
        pdb = ParameterFile(parameter_file, metadata=prms_meta)

        expected_headers = ['Written by Bandit version 0.8.7',
                           'ParamDb revision: https///code.usgs.gov/wma/national-iwaas/nhm/nhm-applications/nhm-v1.1-conus/paramdb_v1.1_gridmet_CONUS/commit/1ffad3a9e33473290efaaa472a90b42a12e28e1f']
        assert pdb.headers == expected_headers
        assert not pdb.exists('altw')

    @pytest.mark.parametrize('name, expected', [('tmax_cbh_adj', np.array([[1.157313, -0.855009, -0.793667, -0.793667, 0.984509, 0.974305,
                                                                            0.974305, 0.448011, 0.461038, 0.461038, 1.214274, 1.157313],
                                                                           [3., 1.098661, 0.93269, 0.93269, -0.987916, -0.966681,
                                                                            -0.966681, 3., 3., 3., 3., 3.]], dtype=np.float32)),
                                                ('potet_sublim', np.array([0.501412, 0.501412], dtype=np.float32))])
    def test_read_parameter_file_subset_hru_param(self, pdb_instance, name, expected):
        assert (pdb_instance.get_subset(name, [57873, 57879]) == expected).all()

    @pytest.mark.parametrize('name, expected', [('seg_cum_area', np.array([ 74067.516, 296268.53 , 108523.89 ], dtype=np.float32))])
    def test_read_parameter_file_subset_segment_param(self, pdb_instance, name, expected):
        assert (pdb_instance.get_subset(name, [30114, 30118, 30116]) == expected).all()

    def test_read_parameter_file_with_control(self, datadir, pdb_instance):
        control_file = datadir.join('control.default.bandit')

        prms_meta = MetaData(verbose=True).metadata
        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False, version=5)

        pdb_instance.control = ctl

        expected_modules = {'et_module': 'potet_jh',
                            'precip_module': 'precipitation_hru',
                            'soilzone_module': 'soilzone',
                            'solrad_module': 'ddsolrad',
                            'srunoff_module': 'srunoff_smidx',
                            'strmflow_module': 'muskingum_mann',
                            'temp_module': 'temperature_hru',
                            'transp_module': 'transp_tindex',
                            'basin_module': 'basin',
                            'intcp_module': 'intcp',
                            'obs_module': 'obs',
                            'snow_module': 'snowcomp',
                            'gw_module': 'gwflow'}
        assert pdb_instance.control.modules == expected_modules

        expected_hru_to_seg = {57863: 30119,
                               57864: 30119,
                               57867: 30115,
                               57868: 30118,
                               57869: 30118,
                               57872: 30115,
                               57873: 30113,
                               57874: 30113,
                               57877: 30114,
                               57878: 30116,
                               57879: 30116,
                               57880: 30114,
                               57881: 30117,
                               57882: 30117}
        assert pdb_instance.hru_to_seg == expected_hru_to_seg

        expected_poi_to_seg = {'06469400': 7}
        assert pdb_instance.poi_to_seg == expected_poi_to_seg

        expected_poi_to_seg0 = {'06469400': 6}
        assert pdb_instance.poi_to_seg0 == expected_poi_to_seg0

        expected_seg_to_hru = {30119: [57863, 57864],
                               30115: [57867, 57872],
                               30118: [57868, 57869],
                               30113: [57873, 57874],
                               30114: [57877, 57880],
                               30116: [57878, 57879],
                               30117: [57881, 57882]}
        assert pdb_instance.seg_to_hru == expected_seg_to_hru

        assert pdb_instance.missing_params == {'nhm_deplcrv', 'pref_flow_infil_frac'}

    def test_parameter_outlier_ids(self, pdb_instance):
        """Check that outlier_ids returns the correct NHM ids for the azrh parameter"""
        assert pdb_instance.outlier_ids('azrh') == [30113, 30115, 30117, 30118, 30119]

    @pytest.mark.parametrize('name', ['vhe',
                                      'ssr2gw_exp',
                                      'tmax_index'])
    def test_remove_multiple_parameters(self, pdb_instance, name):
        remove_list = ['vhe', 'ssr2gw_exp', 'tmax_index']

        pdb_instance.remove(remove_list)
        assert not pdb_instance.exists(name)

    # ==========================================
    # Tests for Parameters class
    def test_poi_upstream_hrus(self, pdb_instance):
        assert pdb_instance.poi_upstream_hrus('06469400') == {'06469400': [57863, 57864, 57867, 57868, 57869, 57872, 57873,
                                                                           57874, 57877, 57878, 57879, 57880, 57881, 57882]}

        # Test with all POIs (only one in this case)
        all_pois = pdb_instance.poi_to_seg.keys()
        assert pdb_instance.poi_upstream_hrus(all_pois) == {'06469400': [57863, 57864, 57867, 57868, 57869, 57872, 57873,
                                                                         57874, 57877, 57878, 57879, 57880, 57881, 57882]}

    def test_poi_upstream_segments(self, pdb_instance):
        assert pdb_instance.poi_upstream_segments('06469400') == {'06469400': [30113, 30114, 30115, 30116, 30117, 30118, 30119]}

        # Test with all POIs (only one in this case)
        all_pois = pdb_instance.poi_to_seg.keys()
        assert pdb_instance.poi_upstream_segments(all_pois) == {'06469400': [30113, 30114, 30115, 30116, 30117, 30118, 30119]}

    def test_segment_upstream_segments(self, pdb_instance):
        assert pdb_instance.segment_upstream_segments(30115) == {30115: [30113, 30114, 30115, 30116, 30117, 30118]}

        # Test with all segments
        all_segs = pdb_instance.get('nhm_seg').data
        assert pdb_instance.segment_upstream_segments(all_segs) == {30113: [30113, 30114, 30116, 30117],
                                                                    30114: [30114],
                                                                    30115: [30113, 30114, 30115, 30116, 30117, 30118],
                                                                    30116: [30114, 30116],
                                                                    30117: [30114, 30116, 30117],
                                                                    30118: [30113, 30114, 30116, 30117, 30118],
                                                                    30119: [30113, 30114, 30115, 30116, 30117, 30118, 30119]}

    def test_segment_upstream_hrus(self, pdb_instance):
        assert pdb_instance.segment_upstream_hrus(30115) == {30115: [57867, 57868, 57869, 57872, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882]}

        # Test with all POIs (only one in this case)
        all_segs = pdb_instance.get('nhm_seg').data
        assert pdb_instance.segment_upstream_hrus(all_segs) == {30113: [57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882],
                                                                30114: [57877, 57880],
                                                                30115: [57867, 57868, 57869, 57872, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882],
                                                                30116: [57877, 57878, 57879, 57880],
                                                                30117: [57877, 57878, 57879, 57880, 57881, 57882],
                                                                30118: [57868, 57869, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882],
                                                                30119: [57863, 57864, 57867, 57868, 57869, 57872, 57873, 57874, 57877, 57878, 57879, 57880, 57881, 57882]}

    def test_parameters_update_element(self, pdb_instance):
        assert (pdb_instance.get('melt_look').data == np.array([90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], dtype=np.int32)).all()

        pdb_instance.update_element('melt_look', 57873, 85)
        assert (pdb_instance.get('melt_look').data == np.array([90, 90, 90, 90, 90, 90, 85, 90, 90, 90, 90, 90, 90, 90], dtype=np.int32)).all()

        assert (pdb_instance.get('seg_depth').data == np.array([1.192112, 0.894294, 1.244646, 0.968989, 1.015976, 1.1965, 1.286962], dtype=np.float32)).all()

        pdb_instance.update_element('seg_depth', 30117, 0.25)
        assert (pdb_instance.get('seg_depth').data == np.array([1.192112, 0.894294, 1.244646, 0.968989, 0.25, 1.1965, 1.286962], dtype=np.float32)).all()

        assert pdb_instance.get('albset_sna').data == np.float32(0.05)

        pdb_instance.update_element('albset_sna', 0, 0.07)
        assert pdb_instance.get('albset_sna').data == np.float32(0.07)

    def test_get_dataframe_bad_param(self, pdb_instance):
        """Test that a KeyError is raised when a bad parameter name is passed to get_dataframe"""
        with pytest.raises(KeyError):
            pdb_instance.get_dataframe('bad_param')

    def test_get_dataframe(self, pdb_instance):
        """Test the return of a dataframe with global IDs index for a parameter"""
        df = pdb_instance.get_dataframe('gwflow_coef')
        expected_df = pd.DataFrame({'gwflow_coef': np.array([0.072118, 0.096741, 0.074658, 0.071941, 0.091905], dtype=np.float32),
                                    'nhm_id': np.array([57863, 57864, 57867, 57868, 57869], dtype=np.int32)}).set_index('nhm_id')

        pd.testing.assert_frame_equal(df.head(), expected_df.head())

    def test_get_dataframe_for_global_ids(self, pdb_instance):
        """Test the return of local indices for index when nhm_id or nhm_seg selected"""
        df = pdb_instance.get_dataframe('nhm_id')
        expected_df = pd.DataFrame({'nhm_id': np.array([57863, 57864, 57867, 57868, 57869], dtype=np.int32),
                                    'model_hru_idx': np.array([1, 2, 3, 4, 5], dtype=np.int64)}).set_index('model_hru_idx')

        pd.testing.assert_frame_equal(df.head(), expected_df.head())

        df = pdb_instance.get_dataframe('nhm_seg')
        expected_df = pd.DataFrame({'nhm_seg': np.array([30113, 30114, 30115, 30116, 30117], dtype=np.int32),
                                    'model_seg_idx': np.array([1, 2, 3, 4, 5], dtype=np.int64)}).set_index('model_seg_idx')

        pd.testing.assert_frame_equal(df.head(), expected_df.head())

    def test_get_dataframe_snarea_curve(self, pdb_instance):
        df = pdb_instance.get_dataframe('snarea_curve')
        expected_df = pd.DataFrame({'curve_index': np.array([1], dtype=np.int64),
                                    1: np.array([0.0], dtype=np.float32),
                                    2: np.array([0.22], dtype=np.float32),
                                    3: np.array([0.43], dtype=np.float32),
                                    4: np.array([0.62], dtype=np.float32),
                                    5: np.array([0.77], dtype=np.float32),
                                    6: np.array([0.88], dtype=np.float32),
                                    7: np.array([0.95], dtype=np.float32),
                                    8: np.array([0.99], dtype=np.float32),
                                    9: np.array([1.0], dtype=np.float32),
                                    10: np.array([1.0], dtype=np.float32),
                                    11: np.array([1.0], dtype=np.float32),}).set_index('curve_index')
        expected_df.columns = np.arange(1, 12, dtype=np.int64)

        pd.testing.assert_frame_equal(df, expected_df)

    def test_get_dataframe_no_global(self, pdb_instance):
        """Test return of local indices when global IDs are not available"""
        pdb_instance.remove('nhm_id')
        pdb_instance.remove('nhm_seg')

        df = pdb_instance.get_dataframe('gwflow_coef')
        expected_df = pd.DataFrame({'gwflow_coef': np.array([0.072118, 0.096741, 0.074658, 0.071941, 0.091905], dtype=np.float32),
                                    'model_hru_idx': np.array([1, 2, 3, 4, 5], dtype=np.int64)}).set_index('model_hru_idx')

        pd.testing.assert_frame_equal(df.head(), expected_df.head())
