import pytest

import xml.dom.minidom as minidom
import xml.etree.ElementTree as xmlET

from pyPRMS import ParamDimensions
from pyPRMS import MetaData


@pytest.fixture(scope='class')
def dims_obj():
    """Instantiate Dimensions object"""
    prms_meta = MetaData(verbose=False).metadata
    dims_obj = ParamDimensions(metadata=prms_meta)
    return dims_obj


class TestEmptyParamDimensions:
    """Tests related to the ParamDimensions class"""

    @pytest.mark.parametrize('name', ['nhru',
                                      'nmonths'])
    def test_add_paramdimensions_with_default_size(self, dims_obj, name):
        """Test if adding dimensions without specifying the size
           results in the default size"""
        dims_obj.add(name=name)
        assert dims_obj[name].size == dims_obj.metadata[name].get('default')

    def test_paramdimensions_xml(self, dims_obj):
        str_cmp = '<?xml version="1.0" ?>\n<dimensions>\n    <dimension name="nhru">\n        <position>1</position>\n        <size>1</size>\n    </dimension>\n    <dimension name="nmonths">\n        <position>2</position>\n        <size>12</size>\n    </dimension>\n</dimensions>\n'
        xmlstr = minidom.parseString(xmlET.tostring(dims_obj.xml)).toprettyxml(indent='    ')
        assert xmlstr == str_cmp


    def test_add_third_paramdimension_exception(self, dims_obj):
        with pytest.raises(ValueError):
            dims_obj.add(name='one')

    @pytest.mark.parametrize('name, position', [('nhru', 0),
                                                ('nmonths', 1)])
    def test_get_paramdimension_index(self, dims_obj, name, position):
        assert dims_obj.get_position(name=name) == position

    @pytest.mark.parametrize('name, position, size', [('nhru', 0, 1),
                                                      ('nmonths', 1, 12)])
    def test_get_dimsize_by_index(self, dims_obj, name, position, size):
        assert dims_obj.get_dimsize_by_index(position) == size

    def test_get_paramdimension_index_bad(self, dims_obj):
        with pytest.raises(IndexError):
            dims_obj.get_dimsize_by_index(2)

    def test_paramdimensions_tostructure(self, dims_obj):
        expected = {'nhru': {'size': 1, 'position': 0}, 'nmonths': {'size': 12, 'position': 1}}

        assert dims_obj.tostructure() == expected
