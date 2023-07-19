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
                                            ('one', 1)])
    def test_add_global_dimensions(self, pdb_instance, name, size):
        pdb_instance.dimensions.add(name=name, size=size)

        assert pdb_instance.dimensions.get(name).size == size

    @pytest.mark.parametrize('name', [('cov_type'),
                                      ('tmax_adj'),
                                      ('basin_solsta')])
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
                                            ('basin_solsta', np.int32(8))])
    def test_parameter_data(self, pdb_instance, name, data):
        pdb_instance[name].data = data

        assert (pdb_instance[name].data == data).all()

    def test_missing_parameter(self, pdb_instance):
        assert not pdb_instance.exists('hru_area')

    def test_get_missing_parameter(self, pdb_instance):
        with pytest.raises(ParameterError):
            aa = pdb_instance.get('nothin')
