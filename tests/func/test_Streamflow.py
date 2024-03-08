import pytest
import os
from distutils import dir_util
import numpy as np
# import xml.dom.minidom as minidom
# import xml.etree.ElementTree as xmlET

from pyPRMS import Streamflow

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

    def test_read_streamflow_data(self, datadir):
        sf_filename = datadir.join('sf_data')

        obs_sf = Streamflow(sf_filename, verbose=False)

        assert obs_sf.data.mean().values[0] == 30.094350205198356
        assert obs_sf.headercount == 10
        assert obs_sf.metaheader == ['ID']
        assert obs_sf.stations == ['06469400']
        assert obs_sf.numdays == 731
        assert obs_sf.units == {'runoff': 'cfs'}
