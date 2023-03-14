import pytest
import numpy as np
from pyPRMS import ControlVariable
from pyPRMS.constants import PTYPE_TO_DTYPE


class TestControlVariable:
    """Tests related to the ControlVariable class"""

    # Add control variable
    # Add value(s) to control variable

    @pytest.mark.parametrize('name, dtype, def_val', [('int_val', 1, 5),
                                                      ('float_val', 2, np.float32(3.4)),
                                                      ('str_val', 4, 'hello')])
    def test_create_control_variable(self, name, dtype, def_val):
        avar = ControlVariable(name, datatype=dtype, default=def_val)

        # A control variable with no value assigned should return the default value
        assert avar.name == name and \
               avar.default == def_val and \
               avar.values == def_val

    @pytest.mark.parametrize('name, dtype, def_val', [('int_val', 1, '5'),
                                                      ('float_val', 2, '5'),
                                                      ('float_val', 2, '3.4')])
    def test_create_control_variable_from_str(self, name, dtype, def_val):
        """Check that number strings are converted to numerics correctly"""
        avar = ControlVariable(name, datatype=dtype, default=def_val)

        # A control variable with no value assigned should return the default value
        assert avar.name == name and \
               avar.default == PTYPE_TO_DTYPE[dtype](def_val) and \
               avar.values == PTYPE_TO_DTYPE[dtype](def_val)

    @pytest.mark.parametrize('name, dtype, def_val', [('int_val', 1, '5.2'),
                                                      ('int_val', 1, 'somestr'),
                                                      ('float_val', 2, 'somestr')])
    def test_create_control_variable_bad_conv_raises(self, name, dtype, def_val):
        """Check that bad conversion from a string raises an error"""

        with pytest.raises(ValueError):
            avar = ControlVariable(name, datatype=dtype, default=def_val)

    @pytest.mark.parametrize('name, dtype, def_val, val', [('int_val', 1, 5, 7),
                                                           ('float_val', 2, np.float32(3.4), 9.2),
                                                           ('str_val', 4, 'hello', 'goodbye')])
    def test_set_value(self, name, dtype, def_val, val):
        avar = ControlVariable(name, datatype=dtype, default=def_val)
        avar.values = val

        assert avar.default != avar.values

    @pytest.mark.parametrize('name, dtype, def_val, val', [('int_val', 1, 5, 7),
                                                           ('float_val', 2, np.float32(3.4), 9.2),
                                                           ('str_val', 4, 'hello', 'goodbye')])
    def test_force_default(self, name, dtype, def_val, val):
        """Always return the default value when force_default is set"""
        avar = ControlVariable(name, datatype=dtype, default=def_val)
        avar.values = val

        assert avar.default != avar.values

        avar.force_default = True
        assert avar.default == avar.values

    @pytest.mark.parametrize('name, dtype, val, ret_type', [('int1', 1, [1, 2, 3, 4], np.ndarray),
                                                            ('int2', 1, 1, np.int32),
                                                            ('str1', 4, 'file1', str),
                                                            ('str2', 4, ['file1', 'file2'], np.ndarray)])
    def test_values_return_type(self, name, dtype, val, ret_type):
        avar = ControlVariable(name, datatype=dtype)
        avar.values = val

        assert type(avar.values) == ret_type
