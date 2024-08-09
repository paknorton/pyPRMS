import pytest
import numpy as np
import pandas as pd
from pyPRMS import Parameters
from pyPRMS.Exceptions_custom import ParameterError, ParameterExistsError
from pyPRMS import MetaData


@pytest.fixture(scope='class')
def pdb_instance():
    prms_meta = MetaData(verbose=False).metadata

    pdb = Parameters(metadata=prms_meta)
    return pdb


class TestParameters:
    # Still to test:
    # - add invalid parameter name

    def test_parameters_read_method_is_abstract(self, pdb_instance):
        """The Parameters class _read() method is abstract"""
        with pytest.raises(AssertionError):
            pdb_instance._read()

    @pytest.mark.parametrize('name, size', [('nhru', 4),
                                            ('nmonths', 12),
                                            ('one', 1),
                                            ('npoigages', 4),
                                            ('nobs', 4)])
    def test_add_global_dimensions(self, pdb_instance, name, size):
        pdb_instance.dimensions.add(name=name, size=size)

        assert pdb_instance.dimensions.get(name).size == size

    @pytest.mark.parametrize('name', [('cov_type'),
                                      ('tmin_cbh_adj'),
                                      ('tmax_adj'),
                                      ('basin_solsta'),
                                      ('poi_gage_id'),
                                      ('poi_gage_segment'),
                                      ('poi_type')])
    def test_add_valid_parameter(self, pdb_instance, name):
        pdb_instance.add(name=name)
        assert pdb_instance.exists(name=name)

    @pytest.mark.parametrize('name', [('cov_type'),
                                      ('tmax_adj'),
                                      ('basin_solsta')])
    def test_add_existing_parameter_error(self, pdb_instance, name):
        # Trying to add a parameter that already exists should raise an error
        with pytest.raises(ParameterExistsError):
            pdb_instance.add(name=name)
    @pytest.mark.parametrize('name, data', [('cov_type', np.array([1, 0, 1, 2], dtype=np.int32)),
                                            ('tmin_cbh_adj', np.array([[1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 11.1, 12.11],
                                                                       [13.12, 14.13, 15.14, 16.15, 17.16, 18.17, 19.18, 20.19, 21.20, 22.21, 23.22, 24.23],
                                                                       [1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 11.1, 12.11],
                                                                       [13.12, 14.13, 15.14, 16.15, 17.16, 18.17, 19.18, 20.19, 21.20, 22.21, 23.22, 24.23]],
                                                                      dtype=np.float32)),
                                            ('tmax_adj', np.zeros(48, dtype=np.float32).reshape((-1, 12), order='F')),
                                            ('basin_solsta', np.int32(8)),
                                            ('poi_gage_id', np.array(['01234567', '12345678', '23456789', '34567890'], dtype=np.str_)),
                                            ('poi_gage_segment', np.array([12, 4, 45, 26], dtype=np.int32)),
                                            ('poi_type', np.array([1, 0, 1, 0], dtype=np.int32))])
    def test_parameter_data(self, pdb_instance, name, data):
        pdb_instance[name].data = data

        assert (pdb_instance[name].data == data).all()

    def test_as_dataframe_2d(self, pdb_instance):
        df = pdb_instance.get('tmin_cbh_adj').as_dataframe
        expected_df = pd.DataFrame({'tmin_cbh_adj_1': [1.0, 13.12, 1.0, 13.12],
                                    'tmin_cbh_adj_2': [2.1, 14.13, 2.1, 14.13],
                                    'tmin_cbh_adj_3': [3.2, 15.14, 3.2, 15.14],
                                    'tmin_cbh_adj_4': [4.3, 16.15, 4.3, 16.15],
                                    'tmin_cbh_adj_5': [5.4, 17.16, 5.4, 17.16],
                                    'tmin_cbh_adj_6': [6.5, 18.17, 6.5, 18.17],
                                    'tmin_cbh_adj_7': [7.6, 19.18, 7.6, 19.18],
                                    'tmin_cbh_adj_8': [8.7, 20.19, 8.7, 20.19],
                                    'tmin_cbh_adj_9': [9.8, 21.20, 9.8, 21.20],
                                    'tmin_cbh_adj_10': [10.9, 22.21, 10.9, 22.21],
                                    'tmin_cbh_adj_11': [11.1, 23.22, 11.1, 23.22],
                                    'tmin_cbh_adj_12': [12.11, 24.23, 12.11, 24.23]}, dtype=np.float32)
        expected_df.rename(index={k: k + 1 for k in expected_df.index}, inplace=True)
        expected_df.index.name = 'model_hru_idx'

        pd.testing.assert_frame_equal(df, expected_df)

    def test_as_dataframe_poi(self, pdb_instance):
        df = pdb_instance.get('poi_gage_id').as_dataframe
        expected_df = pd.DataFrame({'poi_gage_id': ['01234567', '12345678', '23456789', '34567890']}, dtype=np.str_)
        expected_df.rename(index={k: k + 1 for k in expected_df.index}, inplace=True)
        expected_df.index.name = 'model_poi_idx'

        pd.testing.assert_frame_equal(df, expected_df)

    def test_as_dataframe_scalar(self, pdb_instance):
        df = pdb_instance.get('basin_solsta').as_dataframe
        expected_df = pd.DataFrame({'basin_solsta': 8}, index=[1], dtype=np.int32)
        expected_df.index.name = 'idx'

        pd.testing.assert_frame_equal(df, expected_df)

    def test_missing_parameter(self, pdb_instance):
        assert not pdb_instance.exists('hru_area')

    def test_get_missing_parameter(self, pdb_instance):
        with pytest.raises(ParameterError):
            aa = pdb_instance.get('nothin')

    def test_remove_poi_list(self, pdb_instance):
        pdb_instance.remove_poi(['12345678'])
        assert (pdb_instance['poi_gage_id'].data == np.array(['01234567', '23456789', '34567890'], dtype=np.str_)).all()
        assert (pdb_instance['poi_gage_segment'].data == np.array([12, 45, 26], dtype=np.int32)).all()
        assert (pdb_instance['poi_type'].data == np.array([1, 1, 0], dtype=np.int32)).all()
        assert (pdb_instance.dimensions.get('npoigages').size == pdb_instance['poi_gage_id'].data.size)

    def test_remove_poi_str(self, pdb_instance):
        pdb_instance.remove_poi('34567890')
        assert (pdb_instance['poi_gage_id'].data == np.array(['01234567', '23456789'], dtype=np.str_)).all()
        assert (pdb_instance['poi_gage_segment'].data == np.array([12, 45], dtype=np.int32)).all()
        assert (pdb_instance['poi_type'].data == np.array([1, 1], dtype=np.int32)).all()
        assert (pdb_instance.dimensions.get('npoigages').size == pdb_instance['poi_gage_id'].data.size)

    def test_remove_poi_all(self, pdb_instance):
        """When all POIs are removed, the parameters should be removed and the npoigages dimension should be zero."""
        pdb_instance.remove_poi(['01234567', '23456789'])
        assert not pdb_instance.exists('poi_gage_id')
        assert not pdb_instance.exists('poi_gage_segment')
        assert not pdb_instance.exists('poi_type')
        assert (pdb_instance.dimensions.get('npoigages').size == 0)
