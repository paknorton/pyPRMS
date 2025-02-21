import pytest
import os
from pathlib import Path
from distutils import dir_util

from pyPRMS import Cbh
from pyPRMS import ParameterFile
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
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture()
def pdb_instance(datadir):
    parameter_file = datadir / 'myparam.param'

    prms_meta = MetaData(verbose=False).metadata

    pdb = ParameterFile(parameter_file, metadata=prms_meta)
    return pdb


class TestCbh:

    # @pytest.mark.skip(reason="Fixing other problems before testing this")
    def test_read_ascii_roundtrip_ascii(self, datadir, pdb_instance, tmp_path):
        src_file = [str(datadir.join('tmax.day')),
                    str(datadir.join('tmin.day')),
                    str(datadir.join('precip.day'))]

        nhm_ids = pdb_instance.get('nhm_id').data
        cbh = Cbh(src_file, engine='ascii')

        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        for cvar in cbh.data.data_vars:
            # for cvar in ['precip', 'tmax', 'tmin']:
            out_file = out_path / f'{cvar}_chk.day'
            cbh.write_ascii(out_file, variable=cvar)

            with open(datadir.join(f'{cvar}.day'), 'r') as f:
                lines_orig = f.readlines()

            with open(out_file, 'r') as f:
                lines_chk = f.readlines()

            assert lines_orig == lines_chk
