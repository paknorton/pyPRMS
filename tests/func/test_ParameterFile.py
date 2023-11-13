import pytest
import numpy as np
import os
from distutils import dir_util
from pyPRMS import ControlFile
from pyPRMS import ParameterFile
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


