import pytest
import numpy as np
from pyPRMS import ControlVariable
from pyPRMS import MetaData
from pyPRMS.constants import DATA_TYPES, NEW_PTYPE_TO_DTYPE

# from pyPRMS.constants import NEW_PTYPE_TO_DTYPE   # PTYPE_TO_DTYPE

# metadata = {'int_val': {'datatype': 'int32', 'context': 'scalar', 'default': 1, 'force_default': False},
#             'float32_val': {'datatype': 'float32', 'context': 'scalar', 'default': 2.3, 'force_default': False},
#             'str_val': {'datatype': 'string', 'context': 'scalar', 'default': 'none', 'force_default': False},
#             'str_list': {'datatype': 'string', 'context': 'array', 'default': 'none', 'force_default': False}}


@pytest.fixture(scope='class')
def metadata_ctl():
    prms_meta = MetaData(verbose=False).metadata['control']

    return prms_meta


class TestControlVariable:
    """Tests related to the ControlVariable class"""

    # Add control variable
    # Add value(s) to control variable

    # @pytest.mark.parametrize('name', ['int_val',
    #                                   'float32_val',
    #                                   'str_val'])
    @pytest.mark.parametrize('name', ['prms_warmup'])
    def test_create_control_variable(self, metadata_ctl, name):
        avar = ControlVariable(name, meta=metadata_ctl)

        # A control variable with no value assigned should return the default value
        assert avar.name == name and \
               avar.values == avar.meta['default']

    def test_create_control_variable_invalid(self, metadata_ctl):
        with pytest.raises(ValueError):
            avar = ControlVariable('blah', meta=metadata_ctl)

    def test_create_control_variable_nometadata_strict(self):
        with pytest.raises(ValueError):
            avar = ControlVariable('blah')

    def test_create_control_variable_nometadata_nostrict(self):
        avar = ControlVariable('blah', strict=False)
        assert (avar.name == 'blah' and
                avar.values is None and
                avar.size == 0 and
                avar.dyn_param_meaning == [])

    def test_create_control_variable_adhoc_meta_nostrict(self):
        adhoc_meta = {'datatype': 'float32',
                      'description': 'Some cool new variable',
                      'context': 'scalar',
                      'default': 8.2}
        avar = ControlVariable(name='cool_var', meta=adhoc_meta, strict=False)
        assert avar.meta == adhoc_meta and avar.size == 1

    def test_control_variable_str(self, metadata_ctl):
        expected = '----- ControlVariable -----\nname: prms_warmup\nversion: 5.0\ndatatype: int32\ndescription: Number of years to simulate before writing mapped results, Basin, nhru, nsub, or nsegment Summary Output Files\ncontext: scalar\ndefault: 1\n'
        avar = ControlVariable('prms_warmup', meta=metadata_ctl)

        assert avar.__str__() == expected

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
                                           ('initial_deltat', 20.2),
                                           ('start_time', np.array([2000, 2, 8], dtype=np.int32))])
    def test_set_value(self, metadata_ctl, name, val):
        # prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=metadata_ctl)
        avar.values = val

        assert avar.meta['default'] != avar.values
        assert avar.size == 1

    @pytest.mark.parametrize('name, val', [('prms_warmup', [8]),
                                           ('initial_deltat', [20.2, 19.1])])
    def test_set_value_scalar_with_list(self, metadata_ctl, name, val):
        # prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=metadata_ctl)
        avar.values = val

        assert avar.values == NEW_PTYPE_TO_DTYPE[avar.meta['datatype']](val[0])

    @pytest.mark.parametrize('name, val', [('prms_warmup', np.array([8], dtype=np.int32)),
                                           ('initial_deltat', np.array([20.2, 19.1], dtype=np.float32))])
    def test_set_value_scalar_with_array(self, metadata_ctl, name, val):
        # prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=metadata_ctl)
        avar.values = val

        assert avar.values == NEW_PTYPE_TO_DTYPE[avar.meta['datatype']](val[0])

    @pytest.mark.parametrize('name, val', [('prms_warmup', np.array([8], dtype=np.float32)),
                                           ('initial_deltat', np.array([20.2, 19.1], dtype=np.int32))])
    def test_set_value_scalar_with_array_wrong_dtype(self, metadata_ctl, name, val):
        avar = ControlVariable(name, meta=metadata_ctl)

        with pytest.raises(TypeError):
            avar.values = val

    @pytest.mark.parametrize('name, val', [('nsubOutVar_names', ['var1', 'var2', 'var3']),
                                           ('nsubOutVar_names', np.array(['var1', 'var2', 'var3'], dtype=np.str_))])
    def test_set_value_array_with_list_or_array(self, metadata_ctl, name, val):
        # expected = np.array(['var1', 'var2', 'var3'], dtype=np.str_)
        avar = ControlVariable(name, meta=metadata_ctl)
        avar.values = val

        assert np.equal(avar.values, val).all()
        assert avar.size == len(val)

    @pytest.mark.parametrize('name, val', [('prms_warmup', 2.0),
                                           ('initial_deltat', 8),
                                           ('et_module', 1)])
    def test_set_value_scalar_with_scalar_wrong_dtype(self, metadata_ctl, name, val):
        avar = ControlVariable(name, meta=metadata_ctl)

        with pytest.raises(TypeError):
            avar.values = val

    @pytest.mark.parametrize('name, val', [('prms_warmup', 8),
                                           ('et_module', 'potet_hamon'),
                                           ('initial_deltat', 20.2)])
    def test_force_default(self, metadata_ctl, name, val):
        """Always return the default value when force_default is set"""
        # prms_meta = MetaData().metadata['control']

        avar = ControlVariable(name, meta=metadata_ctl)
        avar.values = val

        assert avar.meta['default'] != avar.values

        avar.meta['force_default'] = True
        assert avar.meta['default'] == avar.values

        # The force_default is applied to the global metadata so
        # make sure to set force_default back to False
        avar.meta['force_default'] = False

    @pytest.mark.parametrize('name, expected_meaning', [('prms_warmup', None),
                                                        ('frozen_flag', 'No'),
                                                        ('nhruOutNcol', 'All values for each timestep are written on a single line as in previous versions')])
    def test_value_meaning(self, metadata_ctl, name, expected_meaning):
        avar = ControlVariable(name, meta=metadata_ctl)

        assert avar.value_meaning == expected_meaning

    # @pytest.mark.parametrize('name, val', [('et_module', 'jump')])
    # def test_value_meaning_invalid_value(self, metadata_ctl, name, val):
    #     with pytest.raises(ValueError):
    #         avar = ControlVariable(name, value=val, meta=metadata_ctl)
    #         avar.values = val
    #         assert avar.values == val

    @pytest.mark.parametrize('name, val, expected_meaning', [('nhruOutNcol', 2, 'Number of columns')])
    def test_value_meaning_conditional(self, metadata_ctl, name, val, expected_meaning):
        avar = ControlVariable(name, value=val, meta=metadata_ctl)

        assert avar.value_meaning == expected_meaning
        assert avar.values == val

    @pytest.mark.parametrize('name, val', [('nhruOutON_OFF', 8),
                                           ('et_module', 'blah')])
    def test_crap(self, metadata_ctl, name, val):
        with pytest.raises(ValueError):
            avar = ControlVariable(name=name, value=val, meta=metadata_ctl)
            assert avar.values == val
            aa = avar.value_meaning


    # @pytest.mark.parametrize('name, val, ret_type', [('int_val', 4, np.int32),
    #                                                  ('str1', 'file1', str),
    #                                                  ('str2', ['file1', 'file2'], np.ndarray)])
    # def test_values_return_type(self, name, val, ret_type):
    #     avar = ControlVariable(name, meta=metadata[name])
    #     avar.values = val
    #
    #     assert type(avar.values) == ret_type
