import pytest
import numpy as np
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
                                            ('tmax_adj', np.zeros(48, dtype=np.float32).reshape((-1, 12), order='F')),
                                            ('basin_solsta', np.int32(8)),
                                            ('poi_gage_id', np.array(['01234567', '12345678', '23456789', '34567890'], dtype=np.str_)),
                                            ('poi_gage_segment', np.array([12, 4, 45, 26], dtype=np.int32)),
                                            ('poi_type', np.array([1, 0, 1, 0], dtype=np.int32))])
    def test_parameter_data(self, pdb_instance, name, data):
        pdb_instance[name].data = data

        assert (pdb_instance[name].data == data).all()

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
