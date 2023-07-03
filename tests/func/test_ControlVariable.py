import pytest
# import numpy as np
from pyPRMS import ControlVariable
from pyPRMS import MetaData

# from pyPRMS.constants import NEW_PTYPE_TO_DTYPE   # PTYPE_TO_DTYPE

# metadata = {'int_val': {'datatype': 'int32', 'context': 'scalar', 'default': 1, 'force_default': False},
#             'float32_val': {'datatype': 'float32', 'context': 'scalar', 'default': 2.3, 'force_default': False},
#             'str_val': {'datatype': 'string', 'context': 'scalar', 'default': 'none', 'force_default': False},
#             'str_list': {'datatype': 'string', 'context': 'array', 'default': 'none', 'force_default': False}}


class TestControlVariable:
    """Tests related to the ControlVariable class"""

    # Add control variable
    # Add value(s) to control variable

    # @pytest.mark.parametrize('name', ['int_val',
    #                                   'float32_val',
    #                                   'str_val'])
    @pytest.mark.parametrize('name', ['prms_warmup'])
    def test_create_control_variable(self, name):
        prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=prms_meta[name])

        # A control variable with no value assigned should return the default value
        assert avar.name == name and \
               avar.values == avar.meta['default']

    # @pytest.mark.parametrize('name, dtype, def_val', [('int_val', 1, '5'),
    #                                                   ('float_val', 2, '5'),
    #                                                   ('float_val', 2, '3.4')])
    # def test_create_control_variable_from_str(self, name, dtype, def_val):
    #     """Check that number strings are converted to numerics correctly"""
    #     avar = ControlVariable(name, meta=metadata[name])
    #
    #     # A control variable with no value assigned should return the default value
    #     assert avar.name == name and \
    #            avar.default == PTYPE_TO_DTYPE[dtype](def_val) and \
    #            avar.values == PTYPE_TO_DTYPE[dtype](def_val)

    # @pytest.mark.parametrize('name, dtype, def_val', [('int_val', 1, '5.2'),
    #                                                   ('int_val', 1, 'somestr'),
    #                                                   ('float_val', 2, 'somestr')])
    # def test_create_control_variable_bad_conv_raises(self, name, dtype, def_val):
    #     """Check that bad conversion from a string raises an error"""
    #
    #     with pytest.raises(ValueError):
    #         avar = ControlVariable(name, datatype=dtype, default=def_val)

    @pytest.mark.parametrize('name, val', [('prms_warmup', 8),
                                           ('et_module', 'potet_hamon'),
                                           ('initial_deltat', 20.2)])
    def test_set_value(self, name, val):
        prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=prms_meta[name])
        avar.values = val

        assert avar.meta['default'] != avar.values

    @pytest.mark.parametrize('name, val', [('prms_warmup', 8),
                                           ('et_module', 'potet_hamon'),
                                           ('initial_deltat', 20.2)])
    def test_force_default(self, name, val):
        """Always return the default value when force_default is set"""
        prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=prms_meta[name])
        avar.values = val

        assert avar.meta['default'] != avar.values

        avar.meta['force_default'] = True
        assert avar.meta['default'] == avar.values

    # @pytest.mark.parametrize('name, val, ret_type', [('int_val', 4, np.int32),
    #                                                  ('str1', 'file1', str),
    #                                                  ('str2', ['file1', 'file2'], np.ndarray)])
    # def test_values_return_type(self, name, val, ret_type):
    #     avar = ControlVariable(name, meta=metadata[name])
    #     avar.values = val
    #
    #     assert type(avar.values) == ret_type
