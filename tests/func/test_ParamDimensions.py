import pytest
from pyPRMS import ParamDimensions
from pyPRMS import MetaData


@pytest.fixture(scope='class')
def dims_obj():
    """Instantiate Dimensions object"""
    prms_meta = MetaData(verbose=False).metadata
    dims_obj = ParamDimensions(metadata=prms_meta)
    return dims_obj


class TestEmptyDimensions():
    """Tests related to the ParamDimensions class"""

    @pytest.mark.parametrize('name', ['nhru',
                                      'nmonths'])
    def test_add_dimensions_with_default_size(self, dims_obj, name):
        """Test if adding dimensions without specifying the size
           results in the default size"""
        dims_obj.add(name=name)
        assert dims_obj[name].size == dims_obj.metadata[name].get('default')

    def test_add_third_dimension_exception(self, dims_obj):
        with pytest.raises(ValueError):
            dims_obj.add(name='one')

    @pytest.mark.parametrize('name, position', [('nhru', 0),
                                                ('nmonths', 1)])
    def test_get_dimension_index(self, dims_obj, name, position):
        assert dims_obj.get_position(name=name) == position

    @pytest.mark.parametrize('name, position, size', [('nhru', 0, 1),
                                                      ('nmonths', 1, 12)])
    def test_get_dimsize_by_index(self, dims_obj, name, position, size):
        assert dims_obj.get_dimsize_by_index(position) == size
