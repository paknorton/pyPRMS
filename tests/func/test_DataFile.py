import pytest
import os
from distutils import dir_util
import numpy as np
# import xml.dom.minidom as minidom
# import xml.etree.ElementTree as xmlET

from pyPRMS import DataFile

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
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


class TestStreamflow:

    def test_read_datafile_single_station(self, datadir):
        sf_filename = datadir.join('sf_data_pipestem_bandit')

        obs_sf = DataFile(sf_filename, verbose=False)

        expected_stations = ['06469400']

        assert obs_sf.data.mean().values[0] == 30.094350205198356
        assert len(obs_sf.data.columns) == 1
        assert len(obs_sf.data) == 731
        assert obs_sf.get('runoff')['units'] == 'cfs'
        assert obs_sf.get('runoff')['stations'] == expected_stations


    def test_read_datafile_multiple_stations(self, datadir):
        sf_filename = datadir.join('sf_data_downsizer')

        obs_sf = DataFile(sf_filename, verbose=False)

        expected_mean = {'runoff_14142500': 0.0,
                         'runoff_14137002': 0.0,
                         'runoff_14137000': 2667.07972451021,
                         'runoff_14134000': 178.83350439514788,
                         'runoff_14141500': 341.5708917526521,
                         'runoff_14140000': 1432.5777467542205,
                         'runoff_14139800': 283.73439571085805,
                         'runoff_14139700': 242.826600224096,
                         'runoff_14138850': 889.0568331357849,
                         'runoff_14138800': 222.1890569061978,
                         'runoff_14138720': 0.0,
                         'runoff_14138900': 259.6904311639005,
                         'runoff_14138870': 160.55626150658415,
                         'runoff_14142800': 0.0}

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
        assert obs_sf.get('runoff')['units'] == 'cfs'
        assert obs_sf.get('runoff')['stations'] == expected_stations


    def test_read_datafile_sagehen(self, datadir):
        sf_filename = datadir.join('sagehen.data')

        obs_sf = DataFile(sf_filename, verbose=False)

        # expected_mean = {'runoff_14142500': 0.0,
        #                  'runoff_14137002': 0.0,
        #                  'runoff_14137000': 2667.07972451021,
        #                  'runoff_14134000': 178.83350439514788,
        #                  'runoff_14141500': 341.5708917526521,
        #                  'runoff_14140000': 1432.5777467542205,
        #                  'runoff_14139800': 283.73439571085805,
        #                  'runoff_14139700': 242.826600224096,
        #                  'runoff_14138850': 889.0568331357849,
        #                  'runoff_14138800': 222.1890569061978,
        #                  'runoff_14138720': 0.0,
        #                  'runoff_14138900': 259.6904311639005,
        #                  'runoff_14138870': 160.55626150658415,
        #                  'runoff_14142800': 0.0}
        #
        # expected_stations = ['14142500',
        #                      '14137002',
        #                      '14137000',
        #                      '14134000',
        #                      '14141500',
        #                      '14140000',
        #                      '14139800',
        #                      '14139700',
        #                      '14138850',
        #                      '14138800',
        #                      '14138720',
        #                      '14138900',
        #                      '14138870',
        #                      '14142800']

        # assert obs_sf.data.describe().mean().to_dict() == expected_mean
        assert len(obs_sf.data.columns) == 7   # number of stations
        assert len(obs_sf.data) == 8608   # number of days
        assert list(obs_sf.input_variables.keys()) == ['tmax', 'tmin', 'precip', 'runoff']
        assert obs_sf.get('runoff').get('units') is None
        assert obs_sf.get('runoff').get('stations') is None
