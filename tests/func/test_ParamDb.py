import pytest
import numpy as np
import os
from distutils import dir_util
from pyPRMS import ControlFile
from pyPRMS import ParamDb
from pyPRMS.Exceptions_custom import ControlError
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
    paramdb = datadir.join('paramdb')

    prms_meta = MetaData(verbose=True).metadata

    pdb = ParamDb(paramdb, metadata=prms_meta)
    return pdb


class TestParamDb:

    def test_read_parameter_database(self, pdb_instance):

        assert pdb_instance.dimensions['nhru'].size == 14
        assert pdb_instance.dimensions['nsegment'].size == 7
        assert (pdb_instance.dimensions['npoigages'].size == 1 and
                pdb_instance.dimensions['nobs'].size == 1)

    @pytest.mark.parametrize('name, expected', [('tmax_cbh_adj', np.array([[1.157313, -0.855009, -0.793667, -0.793667, 0.984509, 0.974305,
                                                                            0.974305, 0.448011, 0.461038, 0.461038, 1.214274, 1.157313],
                                                                           [3., 1.098661, 0.93269, 0.93269, -0.987916, -0.966681,
                                                                            -0.966681, 3., 3., 3., 3., 3.]], dtype=np.float32)),
                                                ('potet_sublim', np.array([0.501412, 0.501412], dtype=np.float32)),
                                                ('hru_deplcrv', np.array([1, 1], dtype=np.int32)),
                                                ('snarea_curve', np.array([0., 0.22, 0.43, 0.62, 0.77, 0.88,
                                                                           0.95, 0.99, 1., 1., 1.], dtype=np.float32))])
    def test_read_parameter_database_subset_hru_param(self, pdb_instance, name, expected):
        assert (pdb_instance.get_subset(name, [57873, 57879]) == expected).all()

    @pytest.mark.parametrize('name, expected', [('seg_cum_area', np.array([74067.516, 296268.53, 108523.89],
                                                                          dtype=np.float32))])
    def test_read_parameter_database_subset_segment_param(self, pdb_instance, name, expected):
        assert (pdb_instance.get_subset(name, [30114, 30118, 30116]) == expected).all()

    def test_read_parameter_database_too_many_values(self, datadir):
        paramdb = datadir.join('paramdb_bad')
        prms_meta = MetaData(verbose=True).metadata

        with pytest.raises(IndexError):
            pdb = ParamDb(paramdb, metadata=prms_meta)

