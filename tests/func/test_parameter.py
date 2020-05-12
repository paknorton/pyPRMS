
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

    def test_parameter_dims_gt2_raises(self):
        # Trying to to add more than 2 dimensions to a parameter
        # should raise an error
        aparam = Parameter(name='someparam')
        aparam.dimensions.add(name='nhru', size=4)
        aparam.dimensions.add(name='nmonths', size=12)

        with pytest.raises(ValueError):
            aparam.dimensions.add(name='ngw', size=2)

    def test_add_data_int_to_str(self):
        aparam = Parameter(name='someparam')
        aparam.dimensions.add(name='nhru', size=4)
        aparam.datatype = 4
        aparam.data = [1, 2, 3, 4]

        assert aparam.tolist() == ['1', '2', '3', '4']

    @pytest.mark.parametrize('dtype', [1, 2, 3])
    def test_add_data_bad_convert_raises(self, dtype):
        aparam = Parameter(name='someparam')
        aparam.dimensions.add(name='nhru', size=4)
        aparam.datatype = dtype

        with pytest.raises(ValueError):
            aparam.data = [1, 2, '3a', 4]

    def test_add_bad_datatype_raises(self):
        aparam = Parameter(name='someparam')

        with pytest.raises(TypeError):
            aparam.datatype = 6
