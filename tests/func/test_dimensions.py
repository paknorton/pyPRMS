
import pytest
from pyPRMS import Dimensions
from pyPRMS import MetaData
import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET


@pytest.fixture(scope='class')
def dims_obj():
    """Instantiate Dimensions object"""
    prms_meta = MetaData(verbose=False).metadata
    dims_obj = Dimensions(metadata=prms_meta)
    return dims_obj


class TestEmptyDimensions:
    """Tests related to the Dimension class"""

    def test_dimensions_default_ndims(self, dims_obj):
        """Default dimensions object should have no dimensions"""
        assert dims_obj.ndim == 0

    def test_dimensions_empty_str(self, dims_obj):
        """The __str__ method should produce a pretty string of the object"""
        str_cmp = '<empty>'

        assert dims_obj.__str__() == str_cmp

    def test_dimensions_str(self, dims_obj):
        """The __str__ method should produce a pretty string of the object"""
        str_cmp = "----- Dimension -----\nname: ncascade\ndescription: Number of HRU links for cascading flow\ndefault: 0\nis_fixed: False\nrequires_control: ['cascade_flag > 0']\nsize: 0\n\n"
        dims_obj.add(name='ncascade')

        assert dims_obj.__str__() == str_cmp

    def test_dimensions_xml(self, dims_obj):
        str_cmp = '<?xml version="1.0" ?>\n<dimensions>\n    <dimension name="ncascade">\n        <size>0</size>\n    </dimension>\n</dimensions>\n'
        xmlstr = minidom.parseString(xmlET.tostring(dims_obj.xml)).toprettyxml(indent='    ')
        assert xmlstr == str_cmp

    def test_dimensions_tostructure(self, dims_obj):
        expected = {'ncascade': {'size': 0}}
        assert dims_obj.tostructure() == expected

    @pytest.mark.parametrize('name, size', [('one', 1),
                                            ('nmonths', 12),
                                            ('ndays', 366)])
    def test_dimensions_add_fixed_dimension(self, dims_obj, name, size):
        """Adding fixed dimensions"""
        dims_obj.add(name=name, size=size)
        assert dims_obj[name].size == dims_obj.metadata[name].get('default')

    @pytest.mark.parametrize('name, size', [('seven', 8),
                                            ('nlapse', 2),
                                            ('nglres', 10)])
    def test_fixed_dimension_bad_size_error(self, dims_obj, name, size):
        """Adding a fixed dimension and trying to override the size should
        raise a ValueError
        """
        with pytest.raises(ValueError):
            dims_obj.add(name=name, size=size)

    def test_dimensions_duplicate_ignore(self, dims_obj):
        """Adding a duplicate dimension name should be silently ignored"""
        pre_size = dims_obj.ndim
        pre_exist = dims_obj.exists('one')
        dims_obj.add(name='one', size=1)
        assert pre_exist and dims_obj.ndim == pre_size and dims_obj['one'].size == 1

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

    def test_instantiate_dimensions_no_metadata_strict(self):
        # Strict==True is the default

        with pytest.raises(ValueError):
            dims_obj = Dimensions(strict=True)

    def test_add_bad_dimension(self, dims_obj):
        """Test adding a dimension that is not in the metadata"""

        with pytest.raises(ValueError):
            dims_obj.add(name='random', size=1)

    def test_get_missing_dimension(self, dims_obj):
        """Getting a non-existent dimension should raise an error"""

        with pytest.raises(ValueError):
            dims_obj.get('somedim')
