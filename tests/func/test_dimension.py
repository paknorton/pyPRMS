
import pytest
from pyPRMS import Dimension
from pyPRMS import MetaData


@pytest.fixture(scope='class')
def metadata_instance():
    prms_meta = MetaData(verbose=False).metadata['dimensions']

    return prms_meta


class TestDimension:
    """Tests related to the Dimension class"""

    # def test_create_default_dimension_raises(self):
    #     """A Dimension object with all defaults should raise ValueError"""
    #     with pytest.raises(ValueError):
    #         Dimension()

    def test_create_invalid_dimension_raises(self, metadata_instance):
        """A Dimension object with an invalid name raises ValueError"""
        with pytest.raises(ValueError):
            Dimension(name='baddim', meta=metadata_instance)

    @pytest.mark.parametrize('size', [-1, 1.1, 'a'])
    def test_invalid_size_raises(self, metadata_instance, size):
        with pytest.raises(ValueError):
            Dimension(name='nhru', meta=metadata_instance, size=size)

    def test_create_dimension_adhoc(self, metadata_instance):
        """Create a dimension with adhoc metadata"""
        adim = Dimension(name='nhru',
                         meta={'description': 'Number of HRUs',
                               'size': 1, 'default': 0,
                               'is_fixed': False},
                         strict=False)

    def test_create_dimension_no_metadata_strict(self):
        """Adding an adhoc dimensions with no metadata and strict == True
        should cause an error"""
        with pytest.raises(ValueError):
            adim = Dimension(name='nhru', strict=True)

    def test_create_dimension_no_metadata_nostrict(self):
        """Adding an adhoc dimension with no metadata and strict == False
        should result in a single dimension meta entry for size"""
        adim = Dimension(name='nhru', strict=False)

        assert adim.meta == {'size': 0}

    @pytest.mark.parametrize('size, tname', [(4, 'int'),
                                             ('4', 'str')])
    def test_create_dimension_specified_size(self, metadata_instance, size, tname):
        """Set and get Dimension size"""

        # Set Dimension name and size during instantiation
        adim = Dimension(name='nhru', meta=metadata_instance, size=size)
        assert (adim.name == 'nhru' and adim.size == int(size))

    def test_create_dimension_default_size(self, metadata_instance):
        """Set and get Dimension size"""

        # Set Dimension name during instantiation
        # Set Dimension size after
        adim = Dimension(name='nhru', meta=metadata_instance)
        def_size = adim.size == adim.meta.get('default')

        adim.size = 10
        assert (adim.name == 'nhru' and def_size and adim.size == 10)

    @pytest.mark.parametrize('name, size', [('one', 10),
                                            ('nmonths', 11),
                                            ('ndays', 360)])
    def test_create_dimension_fixed_size_specified(self, metadata_instance, name, size):
        """Certain dimensions have a fixed size and should generate an error
        when created with a different size"""

        # Instantiation with default size
        with pytest.raises(ValueError):
            Dimension(name=name, meta=metadata_instance, size=size)

    @pytest.mark.parametrize('name, size', [('one', 1),
                                            ('nmonths', 12),
                                            ('ndays', 366)])
    def test_increase_dimension_size_fixed(self, metadata_instance, name, size):
        """Dimensions with fixed sizes should not be able to grow in size"""
        # Instantiation with default size

        adim = Dimension(name=name, meta=metadata_instance, size=size)

        # Try adding to the size
        with pytest.raises(ValueError):
            adim += 10

    def test_increase_dimension_size(self, metadata_instance):
        """Non-fixed dimensions should be able to grow in size"""
        adim = Dimension(name='nhru', meta=metadata_instance, size=10)
        adim += 10
        assert adim.size == 20

    def test_increase_dimension_size_wrong_type(self, metadata_instance):
        adim = Dimension(name='nhru', meta=metadata_instance, size=10)

        with pytest.raises(ValueError):
            adim += 10.0

    def test_decrease_dimension_size(self, metadata_instance):
        adim = Dimension(name='nhru', meta=metadata_instance, size=10)
        adim -= 5
        assert adim.size == 5

    def test_decrease_dimension_size_negative(self, metadata_instance):
        adim = Dimension(name='nobs', meta=metadata_instance, size=10)

        with pytest.raises(ValueError):
            adim -= 11

    def test_decrease_dimension_size_less_than_default(self, metadata_instance):
        adim = Dimension(name='nhru', meta=metadata_instance, size=10)

        with pytest.raises(ValueError):
            adim -= 10

    def test_decrease_dimension_size_wrong_type(self, metadata_instance):
        adim = Dimension(name='nhru', meta=metadata_instance, size=10)

        with pytest.raises(ValueError):
            adim -= 5.0

    def test_dimension_repr(self, metadata_instance):
        """The __repr__ should produce code to instantiate a Dimension object"""
        str_cmp = "Dimension(name='nhru', meta={'description': 'Number of HRUs', 'size': 1, 'default': 1, 'is_fixed': False}, size=1, strict=False)"

        adim = Dimension(name='nhru', meta=metadata_instance, size=1)
        repr_str = repr(adim)

        assert repr_str == str_cmp

    def test_dimension_str(self, metadata_instance):
        """The __str__ method should produce a pretty string of the object"""
        str_cmp = '----- Dimension -----\nname: nhru\ndescription: Number of HRUs\ndefault: 1\nis_fixed: False\nsize: 1\n'
        adim = Dimension(name='nhru', meta=metadata_instance, size=1)

        assert adim.__str__() == str_cmp


