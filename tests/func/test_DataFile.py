import pytest
import os
import pandas as pd
import shutil

# import numpy as np
from pandas.testing import assert_frame_equal

from pyPRMS import DataFile
from pyPRMS.metadata.metadata import MetaData
from pyPRMS.parameters.ParameterFile import ParameterFile


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    # 2023-07-18
    # https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)

    return tmpdir

class TestDataFile:

    def test_read_datafile_single_station(self, datadir):
        sf_filename = datadir / 'sf_data_pipestem_bandit'

        expected_str = "----- PRMS data file input variable -----\nname: runoff\ndatatype: float32\ndescription: Streamflow at each measurement station\nunits: runoff_units\nminimum: 0.0\ndimensions: ['nobs']\nmodules: ['muskingum', 'muskingum_lake', 'strmflow_in_out']\n----------\nNumber of rows: 731\nNumber of columns: 1\n"
        prms_meta = MetaData(verbose=False).metadata

        datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)
        obs_sf = datafile.get('runoff')

        expected_stations = ['06469400']

        assert obs_sf.name == 'runoff'
        assert obs_sf.__str__() == expected_str
        assert obs_sf.data.mean().values[0] == 30.094350205198356
        assert len(obs_sf.data.columns) == 1
        assert len(obs_sf.data) == 731
        assert obs_sf.file_units == 'cfs'
        assert list(obs_sf.data.columns) == expected_stations

    # def test_datafile_parameter_units(self, datadir):
    #     sf_filename = datadir / 'sf_data_pipestem_bandit'
    #
    #     prms_meta = MetaData(verbose=False).metadata
    #
    #     datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)
    #     obs_sf = datafile.get('runoff')
    #
    #     assert obs_sf.name == 'runoff'
    #     assert obs_sf.metadata['units'] == 'runoff_units'
    #
    #     sample_derived_units = {'elev_units': 'm', 'precip_units': 'in', 'runoff_units': 'cfs', 'temp_units': 'degF'}
    #     datafile.resolve_units()
    #
    #     assert obs_sf.metadata['units'] == sample_derived_units['runoff_units']

    def test_read_datafile_multiple_stations(self, datadir):
        sf_filename = datadir / 'sf_data_downsizer'

        prms_meta = MetaData(verbose=False).metadata

        datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)
        obs_sf = datafile.get('runoff')

        # A deprecation warning/reminder that checks backwards compatibility
        # until data_by_variable() is completely removed
        with pytest.warns(DeprecationWarning):
            obs_sf_dbv = datafile.data_by_variable("runoff")
        obs_sf_dbv.columns = obs_sf_dbv.columns.str.split("_").str[1]
        pd.testing.assert_frame_equal(obs_sf.data, obs_sf_dbv)
        del obs_sf_dbv

        expected_mean = {'14142500': 0.0,
                         '14137002': 0.0,
                         '14137000': 2667.07972451021,
                         '14134000': 178.83350439514788,
                         '14141500': 341.5708917526521,
                         '14140000': 1432.5777467542205,
                         '14139800': 283.73439571085805,
                         '14139700': 242.826600224096,
                         '14138850': 889.0568331357849,
                         '14138800': 222.1890569061978,
                         '14138720': 0.0,
                         '14138900': 259.6904311639005,
                         '14138870': 160.55626150658415,
                         '14142800': 0.0}

        expected_stations = ['14142500',
                             '14137002',
                             '14137000',
                             '14134000',
                             '14141500',
                             '14140000',
                             '14139800',
                             '14139700',
                             '14138850',
                             '14138800',
                             '14138720',
                             '14138900',
                             '14138870',
                             '14142800']

        assert obs_sf.data.describe().mean().to_dict() == expected_mean
        assert len(obs_sf.data.columns) == 14   # number of stations
        assert len(obs_sf.data) == 731   # number of days
        assert obs_sf.file_units == 'cfs'
        assert list(obs_sf.data.columns) == expected_stations

    def test_read_datafile_sagehen(self, datadir):
        sf_filename = datadir / 'sagehen.data'
        param_filename = datadir / 'sagehen.params'

        prms_meta = MetaData(verbose=False).metadata

        pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)
        datafile = DataFile(sf_filename, metadata=prms_meta, parameters=pdb, verbose=False)
        obs_sf = datafile.get('runoff')

        # assert obs_sf.data.describe().mean().to_dict() == expected_mean
        assert len(datafile.data.columns) == 7   # number of stations
        assert len(datafile.data) == 8608   # number of days
        assert list(datafile.input_variables.keys()) == ['tmax', 'tmin', 'precip', 'runoff']
        assert obs_sf.file_units is None
        # assert obs_sf.get('runoff').get('stations') is None

    @pytest.mark.parametrize('model, missing', [('sagehen', ('-901.0', '-9999.0')),
                                                ('merced', ('-999.0')),
                                                ('boise', ('-999.0')),
                                                ('gulkana', ('-999.0'))])
    def test_roundtrip_datafile(self, datadir, tmp_path, model, missing):
        """Tests reading and writing datafiles

        Original datafile data is compared to the written datafile data.
        The metadata in the written file is not guaranteed to match the
        original datafile.
        """

        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        datafile_filename = datadir / f'{model}.data'
        param_filename = datadir / f'{model}.params'
        chk_filename = out_path / f'{model}_chk.data'

        prms_meta = MetaData(verbose=False).metadata
        pdb = ParameterFile(param_filename, metadata=prms_meta, verbose=False)
        datafile = DataFile(datafile_filename, metadata=prms_meta, parameters=pdb, missing=missing, verbose=False)

        datafile.write_ascii(chk_filename)

        datafile_chk = DataFile(chk_filename, metadata=prms_meta, parameters=pdb, missing=('-999.0'), verbose=False)

        assert_frame_equal(datafile.data, datafile_chk.data, check_dtype=False)

    def test_datafile_repr(self, datadir):
        """Test that DataFile.__repr__ returns a useful string."""
        sf_filename = datadir / 'sf_data_pipestem_bandit'

        prms_meta = MetaData(verbose=False).metadata
        datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)

        result = repr(datafile)
        assert 'DataFile(' in result
        assert 'variables=1' in result
        assert 'period=' in result

    def test_input_variable_repr(self, datadir):
        """Test that InputVariable.__repr__ returns a useful string."""
        sf_filename = datadir / 'sf_data_pipestem_bandit'

        prms_meta = MetaData(verbose=False).metadata
        datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)
        obs_sf = datafile.get('runoff')

        result = repr(obs_sf)
        assert 'InputVariable(' in result
        assert "name='runoff'" in result
        assert 'stations=1' in result
        assert 'rows=731' in result

    def test_invalidate_cache(self, datadir):
        """Test that invalidate_cache forces the data property to rebuild."""
        sf_filename = datadir / 'sf_data_downsizer'

        prms_meta = MetaData(verbose=False).metadata
        datafile = DataFile(sf_filename, metadata=prms_meta, verbose=False)

        # First access builds the cache
        df1 = datafile.data
        assert df1 is datafile.data  # same object returned on second access

        # Invalidate and verify a new frame is built
        datafile.invalidate_cache()
        df2 = datafile.data
        assert df2 is not df1
        assert_frame_equal(df1, df2)
