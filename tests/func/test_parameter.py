
import pytest
from pyPRMS.Parameter import Parameter


class TestParameter:

    def test_create_parameter_defaults(self):
        aparam = Parameter(name='someparam')
        assert aparam.name == 'someparam' and aparam.ndims == 0

    def test_add_data_with_no_dimensions_raises(self):
        aparam = Parameter(name='someparam')

        with pytest.raises(ValueError):
            aparam.data = [1, 2, 3, 4]

    def test_add_data_with_wrong_dtype_raises(self):
        aparam = Parameter(name='someparam')
        aparam.dimensions.add(name='nhru', size=4)
        aparam.datatype = 2

        with pytest.raises(TypeError):
            aparam.data = [1, 2, 3, 4]

    def test_add_bad_datatype_raises(self):
        aparam = Parameter(name='someparam')

        with pytest.raises(TypeError):
            aparam.datatype = 6
