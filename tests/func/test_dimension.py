
import pytest
from pyPRMS.Dimension import Dimension


class TestDimension:
    """Tests related to the Dimension class"""

    # def test_create_default_dimension_raises(self):
    #     """A Dimension object with all defaults should raise ValueError"""
    #     with pytest.raises(ValueError):
    #         Dimension()

    def test_create_invalid_dimension_raises(self):
        """A Dimension object with an invalid name raises ValueError"""
        with pytest.raises(ValueError):
            Dimension(name='baddim')

    @pytest.mark.parametrize('size', [-1, 1.0, 'a'])
    def test_invalid_size_raises(self, size):
        with pytest.raises(ValueError):
            Dimension(name='nhru', size=size)

    @pytest.mark.parametrize('size, tname', [(4, 'int'),
                                             ('4', 'str')])
    def test_create_dimension_specified_size(self, size, tname):
        """Set and get Dimension size"""

        # Set Dimension name and size during instantiation
        adim = Dimension(name='nhru', size=size)
        assert (adim.name == 'nhru' and adim.size == int(size))

    def test_create_dimension_default_size(self):
        """Set and get Dimension size"""

        # Set Dimension name during instantiation
        # Set Dimension size after
        adim = Dimension(name='nhru')
        def_size_zero = adim.size == 0

        adim.size = 10
        assert (adim.name == 'nhru' and def_size_zero and adim.size == 10)

    @pytest.mark.parametrize('name, size, actual_size', [('one', 10, 1),
                                                         ('nmonths', 11, 12),
                                                         ('ndays', 360, 366)])
    def test_create_dimension_fixed_size_specified(self, name, size, actual_size):
        """Certains dimensions should have a fixed size and generate an error
        when created with the wrong size"""

        # Instantiation with default size
        with pytest.raises(ValueError):
            Dimension(name=name, size=size)

        # assert adim.size == actual_size

    @pytest.mark.parametrize('name, size, actual_size', [('one', 1, 1),
                                                         ('nmonths', 12, 12),
                                                         ('ndays', 366, 366)])
    def test_grow_dimension_size_fixed(self, name, size, actual_size):
        """Dimensions with fixed sizes should not be able to grow in size"""
        # Instantiation with default size
        adim = Dimension(name=name, size=size)

        # Try adding to the size
        with pytest.raises(ValueError):
            adim += 10
        # assert adim.size == actual_size

    def test_grow_dimension_size(self):
        """Non-fixed dimensions should be able to grow in size"""
        adim = Dimension(name='nhru', size=10)
        adim += 10
        assert adim.size == 20

