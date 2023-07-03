
import pytest
import numpy as np
from pyPRMS import Parameter
from pyPRMS import MetaData

# @pytest.fixture(scope='class')
# def dims_obj():
#     """Instantiate Dimensions object"""
#     prms_meta = MetaData(verbose=False).metadata
#     dims_obj = Dimensions(metadata=prms_meta)
#     return dims_obj


class TestParameter:

    # Still to test
    # - modified flag set when data is changed

    @pytest.mark.parametrize('name, ndim', [('cov_type', 1),
                                            ('tmax_adj', 2),
                                            ('basin_solsta', 0)])
    def test_create_parameter(self, name, ndim):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
        assert aparam.name == name and aparam.ndim == ndim

    @pytest.mark.parametrize('name, isscalar', [('cov_type', False),
                                                ('tmax_adj', False),
                                                ('basin_solsta', True)])
    def test_is_scalar(self, name, isscalar):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
        assert aparam.name == name and aparam.is_scalar == isscalar

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([1, 0, 1, 2], dtype=np.int32)),
                                            ('tmax_adj', np.array([[2.0, 1.2, 3.3, 0], [2.2, 8, 4, 9]], dtype=np.float32)),
                                            ('basin_solsta', np.int32(8))])
    def test_new_param_data(self, name, data):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
        aparam.data = data

        assert (aparam.data == data).all()

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
    def test_param_change_data_wrong_shape(self, name, data, new_data):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        with pytest.raises(IndexError):
            aparam = Parameter(name=name, meta=prms_meta)
            aparam.data = data
            aparam.data = new_data

    @pytest.mark.parametrize('name, data, new_data', [('basin_solsta',
                                                       np.int32(8),
                                                       np.array([1, 0, 1, 2], dtype=np.int32))])
    def test_param_scalar_data_wrong_class(self, name, data, new_data):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        # Test first time assignment of data
        with pytest.raises(ValueError):
            aparam = Parameter(name=name, meta=prms_meta)
            aparam.data = new_data

        # Test changing existing data
        with pytest.raises(ValueError):
            aparam = Parameter(name=name, meta=prms_meta)
            aparam.data = data
            aparam.data = new_data

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 0, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32))])
    def test_new_param_all_values_equal(self, name, data):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
        aparam.data = data

        assert aparam.all_equal()

    @pytest.mark.parametrize('name, data', [('cov_type', np.array([0, 1, 0, 0], dtype=np.int32)),
                                            ('tmax_adj', np.array([[1, 2, 1, 1], [1, 1, 1, 1]], dtype=np.float32))])
    def test_new_param_all_values_not_equal(self, name, data):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
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
    def test_param_check_dim_type(self, name, ishru, isseg, ispoi):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)

        assert (aparam.is_hru_param() == ishru and aparam.is_seg_param() == isseg and
                aparam.is_poi_param() == ispoi)

    @pytest.mark.parametrize('name, data, under, over', [('cov_type',
                                                          np.array([0, 1, 5, 0], dtype=np.int32),
                                                          0, 1),
                                                         ('tmax_adj',
                                                          np.array([[1, -11, 1, 1], [1, 1, 15, 1]], dtype=np.float32),
                                                          1, 1),
                                                         ('lapsemax_max',
                                                          np.array([1, 2, 1, 3], dtype=np.float32),
                                                          0, 0)])
    def test_param_outliers(self, name, data, under, over):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
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
    def test_param_unique_vals(self, name, data, unique_vals):
        prms_meta = MetaData(verbose=False).metadata['parameters']

        aparam = Parameter(name=name, meta=prms_meta)
        aparam.data = data

        assert (aparam.unique() == unique_vals).all()
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
