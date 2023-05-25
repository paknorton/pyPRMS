
import pytest
from pyPRMS import Dimensions
from pyPRMS import MetaData


@pytest.fixture(scope='class')
def dims_obj():
    """Instantiate Dimensions object"""
    prms_meta = MetaData(verbose=False).metadata
    dims_obj = Dimensions(metadata=prms_meta)
    return dims_obj


class TestEmptyDimensions():
    """Tests related to the Dimension class"""

    def test_dimensions_default_ndims(self, dims_obj):
        """Default dimensions object should have no dimensions"""
        assert dims_obj.ndims == 0

    @pytest.mark.parametrize('name, size', [('one', 1),
                                            ('nmonths', 12),
                                            ('ndays', 366)])
    def test_dimensions_add_fixed_dimension(self, dims_obj, name, size):
        """Adding fixed dimensions"""
        dims_obj.add(name=name, size=size)
        assert dims_obj[name].size == dims_obj.metadata[name].get('default')

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

    @pytest.mark.parametrize('name', ['nlapse',
                                      'nhru'])
    def test_add_dimensions_with_default_size(self, dims_obj, name):
        """Test if adding dimensions without specifying the size
           results in the default size"""
        dims_obj.add(name=name)
        assert dims_obj[name].size == dims_obj.metadata[name].get('default')

    def test_add_bad_dimension(self, dims_obj):
        """Test adding a dimension that is not in the metadata"""

        with pytest.raises(ValueError):
            dims_obj.add(name='random', size=1)