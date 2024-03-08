
import pytest
import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS import Dimensions
from pyPRMS import Parameter
from pyPRMS import MetaData

# @pytest.fixture(scope='class')
# def dims_obj():
#     """Instantiate Dimensions object"""
#     prms_meta = MetaData(verbose=False).metadata
#     dims_obj = Dimensions(metadata=prms_meta)
#     return dims_obj
@pytest.fixture(scope='class')
def metadata_instance():
    prms_meta = MetaData(verbose=False).metadata['parameters']

    return prms_meta


class TestParameter:
    # Still to test
    # - modified flag set when data is changed

    @pytest.mark.parametrize('name, ndim', [('cov_type', 1),
                                            ('tmax_adj', 2),
                                            ('basin_solsta', 0)])
    def test_create_parameter(self, metadata_instance, name, ndim):
        # prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=metadata_instance)
        assert aparam.name == name and aparam.ndim == ndim

    def test_create_parameter_bad(self, metadata_instance):
        """Add parameter which does not exist in metadata"""
        with pytest.raises(ValueError):
            aparam = Parameter(name='someparam', meta=metadata_instance)

    def test_create_parameter_adhoc(self):
        """Add a parameter using adhoc metadata"""
        adhoc_meta = {'datatype': 'float32',
                      'description': 'Awesome undocumented parameter',
                      'units': 'inches',
                      'default': 0.0,
                      'minimum': 0.0,
                      'maximum': 26.2,
                      'dimensions': ['nhru'],
                      'modules': ['awesome_mod']}
        aparam = Parameter(name='awesome_param', meta=adhoc_meta, strict=False)

        assert aparam.meta == adhoc_meta
        assert aparam.ndim == 0
        assert aparam.is_hru_param()
        assert not aparam.is_seg_param()
        assert not aparam.is_poi_param()

    def test_create_parameter_no_metadata_strict(self):
        """A new parameter with no supplied metadata should have an empty dictionary
        for metadata"""
        # When strict == True (default) a parameter must exist in the supplied metadata
        with pytest.raises(ValueError):
            aparam = Parameter(name='someparam', strict=True)

    def test_create_parameter_no_metadata_nostrict(self):
        # When strict == False we can add adhoc parameters
        aparam = Parameter(name='someparam', strict=False)
        assert aparam.meta == {}
        assert aparam.modules == []
        assert aparam.ndim == 0
        assert not aparam.is_hru_param()
        assert not aparam.is_seg_param()
        assert not aparam.is_poi_param()

    @pytest.mark.parametrize('name, isscalar', [('cov_type', False),
                                                ('tmax_adj', False),
                                                ('basin_solsta', True)])
    def test_is_scalar(self, metadata_instance, name, isscalar):
        aparam = Parameter(name=name, meta=metadata_instance)
        assert aparam.name == name and aparam.is_scalar == isscalar

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([1, 0, 1, 2], dtype=np.int32)),
                                            ('tmax_adj', np.array([[2.0, 1.2, 3.3, 0], [2.2, 8, 4, 9]], dtype=np.float32)),
                                            ('basin_solsta', np.int32(8))])
    def test_new_param_data(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert (aparam.data == data).all()
        assert aparam.has_correct_size()

    def test_new_param_no_data(self, metadata_instance):
        """Getting parameter data when data is None raises ValueError"""
        aparam = Parameter(name='tmax_adj', meta=metadata_instance)
        with pytest.raises(ValueError):
            _ = aparam.data

    @pytest.mark.parametrize('name, data, expected', [('cov_type', np.array([1.4, 0, 1.6, 2.1], dtype=np.float32), np.array([1, 0, 1, 2], dtype=np.int32)),
                                                      ('tmax_adj', np.array([[2, 1, 3, 0], [2, 8, 4, 9]], dtype=np.int32), np.array([[2.0, 1.0, 3.0, 0], [2.0, 8, 4, 9]], dtype=np.float32)),
                                                      ('basin_solsta', np.float32(8.2), np.int32(8))])
    def test_new_param_data_cast(self, metadata_instance, name, data, expected):
        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert (aparam.data == expected).all()

    @pytest.mark.parametrize('name, data, expected', [('basin_solsta', np.array([2], dtype=np.int32), 2),
                                                      ('basin_solsta', np.array([3.0], dtype=np.float32), 3),
                                                      ('basin_solsta', np.array([2.5], dtype=np.float32), 2)])
    def test_new_param_data_scalar_given_array(self, metadata_instance, name, data, expected):
        aparam = Parameter(name=name, meta=metadata_instance)

        aparam.data = data
        assert aparam.data == expected

    def test_new_param_data_scalar_given_array_too_big(self, metadata_instance):
        aparam = Parameter(name='basin_solsta', meta=metadata_instance)

        with pytest.raises(IndexError):
            aparam.data = np.array([1, 2], dtype=np.int32)

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([1], dtype=np.int32)),
                                            ('tmax_adj', np.array([2.3], dtype=np.float32))])
    def test_new_param_data_expand(self, metadata_instance, name, data):
        """If a single array value is given it should be expanded to the full size
        of the parameter as supplied in the size of the global dimensions"""
        global_dimensions = Dimensions(metadata=MetaData(verbose=False).metadata)
        global_dimensions.add(name='nhru', size=200)
        global_dimensions.add(name='nmonths', size=12)

        aparam = Parameter(name=name, meta=metadata_instance, global_dims=global_dimensions)
        aparam.data = data

        expected_size = 1
        for xx in aparam.meta['dimensions']:
            expected_size *= global_dimensions.get(xx).size

        assert aparam.data.size == expected_size

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([[1, 0, 1, 2], [1, 1, 2, 2]], dtype=np.int32)),
                                            ('tmax_adj', np.array([2.0, 1.2, 3.3, 0, 8, 4, 9], dtype=np.float32))])
    def test_new_param_data_wrong_ndim(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)

        with pytest.raises(IndexError):
            aparam.data = data

    # @pytest.mark.parametrize('name, data', [('cov_type', np.array([1.0], dtype=np.float32)),
    #                                         ('tmax_adj', np.array([2], dtype=np.int32))])
    # def test_new_param_data_wrong_type(self, metadata_instance, name, data):
    #     with pytest.raises(TypeError):
    #         aparam = Parameter(name=name, meta=metadata_instance)
    #         aparam.data = data

    # TODO: 2023-06-28 PAN: not sure how to handle this
    # @pytest.mark.parametrize('name, data', [('cov_type', np.array([1.2, 0, 1, 2], dtype=np.float32)),
    #                                         ('tmax_adj', np.array([[2.0, 1.2, 3.3, 0], [2.2, 8, 4, 9]], dtype=np.int32)),
    #                                         ('basin_solsta', np.float32(8.2))])
    # def test_new_param_wrong_dtype(self, name, data):
    #     prms_meta = MetaData(verbose=False).metadata['parameters']
    #
    #     with pytest.raises(ValueError):
    #         aparam = Parameter(name=name, meta=prms_meta)
    #         aparam.data = data

    @pytest.mark.parametrize('name, data, new_data', [('cov_type',
                                                       np.array([1, 0, 1, 2], dtype=np.int32),
                                                       np.array([[1, 0, 1], [1, 1, 1]], dtype=np.int32)),])
                                                      # ('tmax_adj',
                                                      #  np.array([[2.0, 1.2, 3.3, 0], [2.2, 8, 4, 9]], dtype=np.float32),
                                                      #  np.array([[2.0, 1.2], [3.3, 0], [2.2, 8], [4, 9]], dtype=np.float32))])
    def test_param_change_data_wrong_shape(self, metadata_instance, name, data, new_data):
        # prms_meta = MetaData(verbose=False).metadata['parameters']

        with pytest.raises(IndexError):
            aparam = Parameter(name=name, meta=metadata_instance)
            aparam.data = data
            aparam.data = new_data

    @pytest.mark.parametrize('name, data, new_data', [('basin_solsta',
                                                       np.int32(8),
                                                       np.array([1, 0, 1, 2], dtype=np.int32))])
    def test_param_scalar_data_wrong_class(self, metadata_instance, name, data, new_data):
        # Test first time assignment of data
        with pytest.raises(IndexError):
            aparam = Parameter(name=name, meta=metadata_instance)
            aparam.data = new_data

        # Test changing existing data
        with pytest.raises(IndexError):
            aparam = Parameter(name=name, meta=metadata_instance)
            aparam.data = data
            aparam.data = new_data

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 0, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32)),
                                            ('temp_units', np.int32(0))])
    def test_new_param_all_values_equal(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert aparam.all_equal()

    def test_new_param_all_values_equal_no_data_raises(self, metadata_instance):
        """If a parameter has no data then all_equal() raises a ValueError"""
        aparam = Parameter(name='tmax_adj', meta=metadata_instance)

        with pytest.raises(ValueError):
            aparam.all_equal()

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 1, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 2, 1, 1], [1, 1, 1, 1]], dtype=np.float32))])
    def test_new_param_all_values_not_equal(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert not aparam.all_equal()

    @pytest.mark.parametrize('name, ishru, isseg, ispoi', [('cov_type', True, False, False),
                                                           ('tmax_adj', True, False, False),
                                                           ('poi_gage_segment', False, False, True),
                                                           ('poi_gage_id', False, False, True),
                                                           ('poi_type', False, False, True),
                                                           ('seg_elev', False, True, False),
                                                           ('seg_humidity', False, True, False),
                                                           ('basin_solsta', False, False, False)])
    def test_param_check_dim_type(self, metadata_instance, name, ishru, isseg, ispoi):
        aparam = Parameter(name=name, meta=metadata_instance)

        assert (aparam.is_hru_param() == ishru and aparam.is_seg_param() == isseg and
                aparam.is_poi_param() == ispoi)

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 1, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 2, 1, 1], [1, 1, 1, 1]], dtype=np.float32))])
    def test_param_check_values(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert aparam.check_values()

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 1, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 2, 1, 1], [1, 1, 1, 1]], dtype=np.float32))])
    def test_param_check_values_no_data_raises(self, metadata_instance, name, data):
        aparam = Parameter(name=name, meta=metadata_instance)
        # aparam.data = data

        with pytest.raises(ValueError):
            assert aparam.check_values()

    @pytest.mark.parametrize('name, data, under, over', [('cov_type',
                                                          np.array([0, 1, 5, 0], dtype=np.int32),
                                                          0, 1),
                                                         ('tmax_adj',
                                                          np.array([[1, -11, 1, 1], [1, 1, 15, 1]], dtype=np.float32),
                                                          1, 1),
                                                         ('lapsemax_max',
                                                          np.array([1, 2, 1, 3], dtype=np.float32),
                                                          0, 0)])
    def test_param_outliers(self, metadata_instance, name, data, under, over):
        # prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        outlier_chk = aparam.outliers()

        assert (outlier_chk.under == under and outlier_chk.over == over)

    @pytest.mark.parametrize('name, data, unique_vals', [('cov_type',
                                                          np.array([0, 1, 5, 0], dtype=np.int32),
                                                          np.array([0, 1, 5], dtype=np.int32)),
                                                         ('tmax_adj',
                                                          np.array([[1, -11, 1, 1], [1, 1, 15, 1]], dtype=np.float32),
                                                          np.array([-11, 1, 15], dtype=np.float32)),
                                                         ('lapsemax_max',
                                                          np.array([1, 2, 1, 3], dtype=np.float32),
                                                          np.array([1, 2, 3], dtype=np.float32))])
    def test_param_unique_vals(self, metadata_instance, name, data, unique_vals):
        # prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=metadata_instance)
        aparam.data = data

        assert (aparam.unique() == unique_vals).all()

    def test_param_dunder_str(self, metadata_instance):
        expected_str = "----- Parameter -----\nname: tmax_adj\ndatatype: float32\ndescription: HRU maximum temperature adjustment\nhelp: Adjustment to maximum temperature for each HRU, estimated on the basis of slope and aspect\nunits: temp_units\ndefault: 0.0\nminimum: -10.0\nmaximum: 10.0\ndimensions: ['nhru', 'nmonths']\nmodules: ['temp_1sta', 'temp_sta', 'temp_laps', 'temp_dist2', 'ide_dist', 'xyz_dist']\n"

        aparam = Parameter(name='tmax_adj', meta=metadata_instance)
        assert aparam.__str__() == expected_str

    def test_param_xml(self, metadata_instance):
        expected = '<?xml version="1.0" ?>\n<parameter name="tmax_adj" version="ver">\n<dimensions>\n<dimension name="nhru">\n<position>1</position>\n<size>0</size>\n</dimension>\n<dimension name="nmonths">\n<position>2</position>\n<size>0</size>\n</dimension>\n</dimensions>\n</parameter>\n'
        aparam = Parameter(name='tmax_adj', meta=metadata_instance)
        xmlstr = minidom.parseString(xmlET.tostring(aparam.xml)).toprettyxml(indent='')
        assert xmlstr == expected

    # def test_add_data_with_no_dimensions_raises(self):
    #     aparam = Parameter(name='someparam')
    #
    #     with pytest.raises(ValueError):
    #         aparam.data = [1, 2, 3, 4]
    #
    # def test_parameter_dims_gt2_raises(self):
    #     # Trying to to add more than 2 dimensions to a parameter
    #     # should raise an error
    #     aparam = Parameter(name='someparam')
    #     aparam.dimensions.add(name='nhru', size=4)
    #     aparam.dimensions.add(name='nmonths', size=12)
    #
    #     with pytest.raises(ValueError):
    #         aparam.dimensions.add(name='ngw', size=2)
    #
    # def test_add_data_int_to_str(self):
    #     # Integer data added to a string parameter should be converted
    #     # to an array of strings.
    #     aparam = Parameter(name='someparam')
    #     aparam.dimensions.add(name='nhru', size=4)
    #     aparam.datatype = 4
    #     aparam.data = [1, 2, 3, 4]
    #
    #     assert aparam.tolist() == ['1', '2', '3', '4']
    #
    # @pytest.mark.parametrize('dtype', [1, 2, 3])
    # def test_add_data_bad_convert_raises(self, dtype):
    #     # Trying to add string data to numeric datatypes should cause an error
    #     aparam = Parameter(name='someparam')
    #     aparam.dimensions.add(name='nhru', size=4)
    #     aparam.datatype = dtype
    #
    #     with pytest.raises(ValueError):
    #         aparam.data = [1, 2, '3a', 4]
    #
    # def test_add_bad_datatype_raises(self):
    #     aparam = Parameter(name='someparam')
    #
    #     with pytest.raises(TypeError):
    #         aparam.datatype = 6
