
import pytest
from pyPRMS.Dimensions import Dimensions


@pytest.fixture(scope='class')
def dims_obj():
    """Instantiate Dimensions object"""
    dims_obj = Dimensions()
    return dims_obj


class TestEmptyDimensions():
    """Tests related to the Dimension class"""

    def test_dimensions_default_ndims(self, dims_obj):
        """Default dimensions object should have no dimensions"""
        assert dims_obj.ndims == 0

    @pytest.mark.parametrize('name, size, actual_size', [('one', 10, 1),
                                                         ('nmonths', 11, 12),
                                                         ('ndays', 360, 366)])
    def test_dimensions_add_fixed_dimension(self, dims_obj, name, size, actual_size):
        dims_obj.add(name=name, size=size)
        assert dims_obj[name].size == actual_size
        pass

    def test_dimensions_duplicate_ignore(self, dims_obj):
        """Adding a duplicate dimension name should be silently ignored"""
        pre_size = dims_obj.ndims
        pre_exist = dims_obj.exists('one')
        dims_obj.add(name='one', size=1)
        assert pre_exist and dims_obj.ndims == pre_size and dims_obj['one'].size == 1

    def test_dimensions_remove(self, dims_obj):
        """Test removing a dimension"""

        pre_exist = dims_obj.exists('one')
        dims_obj.remove('one')
        assert pre_exist and not dims_obj.exists('one')