import pytest
import os
import numpy as np
import xarray as xr
from pathlib import Path
import shutil

from pyPRMS import Cbh
from pyPRMS import ControlFile
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
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)

    return tmpdir

@pytest.fixture()
def meta_instance():
    return MetaData(verbose=False)

@pytest.fixture()
def pdb_instance(datadir, meta_instance):
    parameter_file = datadir / 'myparam.param'

    prms_meta = meta_instance.metadata

    pdb = ParameterFile(parameter_file, metadata=prms_meta)
    return pdb


class TestCbh:

    # @pytest.mark.skip(reason="Fixing other problems before testing this")
    def test_read_ctl_ascii_roundtrip_ascii(self, datadir, pdb_instance, meta_instance, tmp_path):
        # src_file = [str(datadir.join('tmax.day')),
        #             str(datadir.join('tmin.day')),
        #             str(datadir.join('precip.day'))]

        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        nhm_ids = pdb_instance.get('nhm_id').data

        ctl = ControlFile(datadir / 'control.default.bandit', metadata=meta_instance.metadata, verbose=False)
        cbh = Cbh(str(datadir), engine='ascii', metadata=meta_instance.metadata, control=ctl,
                  parameters=pdb_instance, verbose=True)

        # assert not cbh.has_nhm_id
        # cbh.set_nhm_id(nhm_ids)
        assert cbh.has_nhm_id

        for cvar in cbh.data.data_vars:
            if cvar == 'nhm_id':
                continue

            out_file = out_path / cbh.cbh_src[str(cvar)]
            cbh.write_ascii(out_file, variable=str(cvar))

            with open(datadir.join(cbh.cbh_src[str(cvar)]), 'r') as f:
                lines_orig = f.readlines()

            with open(out_file, 'r') as f:
                lines_chk = f.readlines()

            assert lines_orig == lines_chk

    def test_read_single_ascii_roundtrip_ascii(self, datadir, pdb_instance, meta_instance, tmp_path):
        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        cbh = Cbh(str(datadir.join('tmax.day')), engine='ascii', metadata=meta_instance.metadata)

        for cvar in cbh.data.data_vars:
            out_file = out_path / cbh.cbh_src[str(cvar)]
            cbh.write_ascii(out_file, variable=str(cvar))

            with open(datadir.join(cbh.cbh_src[str(cvar)]), 'r') as f:
                lines_orig = f.readlines()

            with open(out_file, 'r') as f:
                lines_chk = f.readlines()

            assert lines_orig == lines_chk

    def test_read_netcdf_roundtrip_netcdf(self, datadir, pdb_instance, meta_instance, tmp_path):
        out_path = tmp_path / 'run_files'
        out_path.mkdir()

        cbh = Cbh(str(datadir.join('cbh.nc')), engine='netcdf', metadata=meta_instance.metadata, parameters=pdb_instance)

        out_file = out_path / 'cbh.nc'
        cbh.write_netcdf(out_file)

        # Check that the values of the data variables match
        ds_tmp = xr.open_dataset(out_file, chunks={})
        ds_tmp = ds_tmp.assign_coords(nhru=ds_tmp.nhm_id)

        ds_orig = cbh.data

        for vv in ds_orig.data_vars:
            np.testing.assert_equal(ds_orig[vv].values, ds_tmp[vv].values)
