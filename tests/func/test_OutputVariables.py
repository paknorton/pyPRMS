import pytest
import os
import shutil

from collections import defaultdict

from pyPRMS.summary.OutputVariables import OutputVariables
from pyPRMS import ControlFile
from pyPRMS.metadata.metadata import MetaData

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


class TestOutputVariables:

    def test_read_output_variables(self, datadir):
        control_file = datadir / 'control.default'
        prms_meta = MetaData(verbose=False).metadata
        ctl = ControlFile(control_file, metadata=prms_meta, verbose=False)
        ovd = OutputVariables(ctl, prms_meta, datadir)

        assert ovd.available_vars == {'hru_outflow': f'{datadir}/nhru_hru_outflow.csv',
                                      'hru_actet': f'{datadir}/nhru_hru_actet.csv',
                                      'hru_ppt': f'{datadir}/nhru_hru_ppt.csv',
                                      'seg_outflow': f'{datadir}/nsegment_seg_outflow.csv',
                                      'basin_infil': f'{datadir}/basin_summary.csv',
                                      'basin_cfs': f'{datadir}/basin_summary.csv',
                                      'basin_tmax': f'{datadir}/basin_summary.csv',
                                      'basin_tmin': f'{datadir}/basin_summary.csv',
                                      'basin_rain': f'{datadir}/basin_summary.csv',
                                      'basin_snow': f'{datadir}/basin_summary.csv'}

        cvar = ovd.get('hru_ppt')
        assert cvar.metadata == defaultdict(list, {'datatype': 'float32',
                                                   'description': 'Precipitation distributed to each HRU',
                                                   'units': 'inches',
                                                   'dimensions': ['nhru'],
                                                   'modules': ['precipitation']})

    