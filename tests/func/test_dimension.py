
import pytest
from pyPRMS import Dimension
from pyPRMS import MetaData


class TestDimension:
    """Tests related to the Dimension class"""

    # def test_create_default_dimension_raises(self):
    #     """A Dimension object with all defaults should raise ValueError"""
    #     with pytest.raises(ValueError):
    #         Dimension()

    def test_create_invalid_dimension_raises(self):
        """A Dimension object with an invalid name raises ValueError"""
        prms_meta = MetaData().metadata['dimensions']

        with pytest.raises(ValueError):
            Dimension(name='baddim', meta=prms_meta)

    @pytest.mark.parametrize('size', [-1, 1.0, 'a'])
    def test_invalid_size_raises(self, size):
        prms_meta = MetaData().metadata['control']

        with pytest.raises(ValueError):
            Dimension(name='nhru', meta=prms_meta, size=size)

    @pytest.mark.parametrize('size, tname', [(4, 'int'),
                                             ('4', 'str')])
    def test_create_dimension_specified_size(self, size, tname):
        """Set and get Dimension size"""

        prms_meta = MetaData().metadata['dimensions']

        # Set Dimension name and size during instantiation
        adim = Dimension(name='nhru', meta=prms_meta, size=size)
        assert (adim.name == 'nhru' and adim.size == int(size))

    def test_create_dimension_default_size(self):
        """Set and get Dimension size"""

        # Set Dimension name during instantiation
        # Set Dimension size after
        prms_meta = MetaData().metadata['dimensions']

        adim = Dimension(name='nhru', meta=prms_meta)
        def_size = adim.size == adim.meta.get('default')

        adim.size = 10
        assert (adim.name == 'nhru' and def_size and adim.size == 10)

    @pytest.mark.parametrize('name, size', [('one', 10),
                                            ('nmonths', 11),
                                            ('ndays', 360)])
    def test_create_dimension_fixed_size_specified(self, name, size):
        """Certain dimensions have a fixed size and should generate an error
        when created with a different size"""

        prms_meta = MetaData().metadata['dimensions']

        # Instantiation with default size
        with pytest.raises(ValueError):
            Dimension(name=name, meta=prms_meta, size=size)

    @pytest.mark.parametrize('name, size', [('one', 1),
                                            ('nmonths', 12),
                                            ('ndays', 366)])
    def test_increase_dimension_size_fixed(self, name, size):
        """Dimensions with fixed sizes should not be able to grow in size"""
        # Instantiation with default size
        prms_meta = MetaData().metadata['dimensions']

        adim = Dimension(name=name, meta=prms_meta, size=size)

        # Try adding to the size
        with pytest.raises(ValueError):
            adim += 10

    def test_increase_dimension_size(self):
        """Non-fixed dimensions should be able to grow in size"""
        prms_meta = MetaData().metadata['dimensions']
        adim = Dimension(name='nhru', meta=prms_meta, size=10)
        adim += 10
        assert adim.size == 20

    def test_decrease_dimension_size(self):
        prms_meta = MetaData().metadata['dimensions']

        adim = Dimension(name='nhru', meta=prms_meta, size=10)
        adim -= 5
        assert adim.size == 5

    def test_decrease_dimension_size_negative(self):
        prms_meta = MetaData().metadata['dimensions']

        adim = Dimension(name='nobs', meta=prms_meta, size=10)

        with pytest.raises(ValueError):
            adim -= 11

    def test_decrease_dimension_size_less_than_default(self):
        prms_meta = MetaData().metadata['dimensions']

        adim = Dimension(name='nhru', meta=prms_meta, size=10)

        with pytest.raises(ValueError):
            adim -= 10
