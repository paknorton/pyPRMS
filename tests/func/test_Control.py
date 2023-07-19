import pytest
import numpy as np
from pyPRMS import Control
from pyPRMS import MetaData
from pyPRMS.Exceptions_custom import ControlError


@pytest.fixture(scope='class')
def control_object():
    prms_meta = MetaData(verbose=False).metadata
    ctl = Control(metadata=prms_meta)

    return ctl


@pytest.fixture(scope='class')
def metadata_ctl():
    prms_meta = MetaData(verbose=False).metadata['control']

    return prms_meta


class TestControl:

    def test_getitem(self, control_object):
        assert control_object['print_debug'].values == 0

    def test_control_read_method_is_abstract(self, control_object):
        """The Control class _read() method is abstract"""
        with pytest.raises(AssertionError):
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
        assert control_object.additional_modules == ['subbasin']

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
            control_object.add(name=name, meta=metadata_ctl)

    def test_add_duplicate_variable(self, control_object, metadata_ctl):
        """Add a duplicate control variable"""

        with pytest.raises(ControlError):
            control_object.add(name='et_module', meta=metadata_ctl)

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