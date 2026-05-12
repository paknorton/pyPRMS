import pytest
import os
import shutil

from pyPRMS.summary.OutputCSV import OutputCSV
from pyPRMS.summary.OutputVariable import OutputVariable

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


class TestOutputCSV:

    # def test_OutputCSV_repr(self, datadir):
    #     """The __repr__ should produce code to instantiate a Dimension object"""
    #     str_cmp = "OutputCSV(filename=PosixPath('./test_OutputCSV/prms_basin_and_streamflow.csv'))"
    #
    #     filename = datadir / 'prms_streamflow.csv'
    #     csv = OutputCSV(filename)
    #
    #     repr_str = repr(csv)
    #     assert repr_str == str_cmp

    def test_read_output_csv_streamflow(self, datadir):
        """Test reading a streamflow output CSV file."""
        filename = datadir / 'prms_streamflow.csv'
        csv = OutputCSV(filename)

        # Check sorted list of variables
        assert csv.variables == ['14144800', '14144900', '14145500', '14146500', '14147500', '14148000',
                                 '14150000', '14150300', '14150800', '14151000', '14152000']

        # Streamflow-only CSV file should have no basin variables
        assert csv.basin_vars == []

        # Check in-order list of POI variables
        assert csv.pois == ['14152000', '14150000', '14148000', '14145500', '14144800', '14144900',
                            '14146500', '14147500', '14151000', '14150800', '14150300']

        # Check poi segment mapping
        assert csv.poi_segments == {'14152000': 0,
                                    '14150000': 2,
                                    '14148000': 4,
                                    '14145500': 7,
                                    '14144800': 10,
                                    '14144900': 15,
                                    '14146500': 18,
                                    '14147500': 22,
                                    '14151000': 26,
                                    '14150800': 29,
                                    '14150300': 31}

        expected_mean = {'14152000': 4199.339247860956,
                         '14150000': 2821.3083179928094,
                         '14148000': 2859.06398898659,
                         '14145500': 1621.567912886303,
                         '14144800': 1322.9002558099905,
                         '14144900': 185.74238225572472,
                         '14146500': 110.11960621369612,
                         '14147500': 1467.7904311929235,
                         '14151000': 1016.5543209330826,
                         '14150800': 284.04695648793336,
                         '14150300': 802.2033224186812}

        assert csv.data.describe().mean().to_dict() == expected_mean

    def test_read_output_csv_basin_and_streamflow(self, datadir):
        """Test reading a basin and streamflow output CSV file."""
        filename = datadir / 'prms_basin_and_streamflow.csv'
        csv = OutputCSV(filename)

        # Check sorted list of variables
        assert csv.variables == ['14144800', '14144900', '14145500', '14146500',
                                 '14147500', '14148000', '14150000', '14150300',
                                 '14150800', '14151000', '14152000',
                                 'basin_actet', 'basin_capwaterin', 'basin_cfs', 'basin_dnflow',
                                 'basin_dprst_evap', 'basin_dprst_seep', 'basin_dprst_volcl',
                                 'basin_dprst_volop', 'basin_dunnian', 'basin_gwflow',
                                 'basin_gwflow_cfs', 'basin_gwin', 'basin_gwsink',
                                 'basin_gwstor', 'basin_gwstor_minarea_wb', 'basin_hortonian',
                                 'basin_imperv_evap', 'basin_imperv_stor', 'basin_intcp_evap',
                                 'basin_intcp_stor', 'basin_lake_stor', 'basin_lakeevap',
                                 'basin_perv_et', 'basin_pk_precip', 'basin_potet',
                                 'basin_ppt', 'basin_pref_flow_in', 'basin_pref_stor',
                                 'basin_prefflow', 'basin_pweqv', 'basin_recharge',
                                 'basin_slowflow', 'basin_slstor', 'basin_snowcov',
                                 'basin_snowevap', 'basin_snowmelt', 'basin_soil_moist',
                                 'basin_soil_rechr', 'basin_soil_to_gw', 'basin_sroff_cfs',
                                 'basin_ssflow_cfs', 'basin_ssstor', 'basin_stflow_in',
                                 'basin_stflow_out', 'basin_surface_storage', 'basin_swrad',
                                 'basin_sz2gw', 'basin_tmax', 'basin_tmin', 'basin_total_storage',
                                 'runoff_cfs']

        # In-order list of basin variables
        assert csv.basin_vars == ['basin_potet', 'basin_actet', 'basin_dprst_evap',
                                  'basin_imperv_evap', 'basin_intcp_evap', 'basin_lakeevap',
                                  'basin_perv_et', 'basin_snowevap', 'basin_swrad',
                                  'basin_ppt', 'basin_pk_precip', 'basin_tmax',
                                  'basin_tmin', 'basin_snowcov', 'basin_total_storage',
                                  'basin_surface_storage', 'basin_dprst_volcl', 'basin_dprst_volop',
                                  'basin_gwstor', 'basin_imperv_stor', 'basin_intcp_stor',
                                  'basin_lake_stor', 'basin_pweqv', 'basin_soil_moist',
                                  'basin_ssstor', 'basin_pref_stor', 'basin_slstor',
                                  'basin_soil_rechr', 'basin_capwaterin', 'basin_dprst_seep',
                                  'basin_gwin', 'basin_pref_flow_in', 'basin_recharge',
                                  'basin_snowmelt', 'basin_soil_to_gw', 'basin_sz2gw',
                                  'basin_gwsink', 'basin_prefflow', 'basin_slowflow',
                                  'basin_hortonian', 'basin_dunnian', 'basin_stflow_in',
                                  'basin_stflow_out', 'basin_gwflow', 'basin_dnflow',
                                  'basin_gwstor_minarea_wb', 'basin_cfs', 'basin_gwflow_cfs',
                                  'basin_sroff_cfs', 'basin_ssflow_cfs', 'runoff_cfs']

        # Check in-order list of variables
        assert csv.pois == ['14152000', '14150000', '14148000',
                            '14145500', '14144800', '14144900',
                            '14146500', '14147500', '14151000',
                            '14150800', '14150300']

        # Check poi segment mapping
        assert csv.poi_segments == {'14152000': 0,
                                    '14150000': 2,
                                    '14148000': 4,
                                    '14145500': 7,
                                    '14144800': 10,
                                    '14144900': 15,
                                    '14146500': 18,
                                    '14147500': 22,
                                    '14151000': 26,
                                    '14150800': 29,
                                    '14150300': 31}

        expected_mean = {'basin_potet': 3.501075027977063,
                         'basin_actet': 3.500971904687407,
                         'basin_dprst_evap': 3.5,
                         'basin_imperv_evap': 3.5,
                         'basin_intcp_evap': 3.500397994390217,
                         'basin_lakeevap': 3.5,
                         'basin_perv_et': 3.5007021496128257,
                         'basin_snowevap': 3.5,
                         'basin_swrad': 92.47589761406203,
                         'basin_ppt': 4.033359467554684,
                         'basin_pk_precip': 3.8913971754149808,
                         'basin_tmax': 35.64606323396385,
                         'basin_tmin': 22.026957799298962,
                         'basin_snowcov': 3.82628264828236,
                         'basin_total_storage': 8.153932581506018,
                         'basin_surface_storage': 3.763018763818769,
                         'basin_dprst_volcl': 3.5,
                         'basin_dprst_volop': 3.525878232394883,
                         'basin_gwstor': 5.984158152789203,
                         'basin_imperv_stor': 3.5,
                         'basin_intcp_stor': 3.531747075945174,
                         'basin_lake_stor': 3.5,
                         'basin_pweqv': 3.7060760551966974,
                         'basin_soil_moist': 4.256027118722028,
                         'basin_ssstor': 4.74127653897104,
                         'basin_pref_stor': 3.5,
                         'basin_slstor': 4.74127653897104,
                         'basin_soil_rechr': 4.041820759969206,
                         'basin_capwaterin': 3.638153807436062,
                         'basin_dprst_seep': 3.500805139815033,
                         'basin_gwin': 3.626873278253564,
                         'basin_pref_flow_in': 3.5,
                         'basin_recharge': 3.626873278253564,
                         'basin_snowmelt': 3.910176890258711,
                         'basin_soil_to_gw': 3.5871332120655883,
                         'basin_sz2gw': 3.53932302789075,
                         'basin_gwsink': 3.5,
                         'basin_prefflow': 3.5,
                         'basin_slowflow': 3.6185516385615926,
                         'basin_hortonian': 3.562538241909218,
                         'basin_dunnian': 3.5,
                         'basin_stflow_in': 3.7194854686677137,
                         'basin_stflow_out': 3.615750369591617,
                         'basin_gwflow': 3.5381882357135943,
                         'basin_dnflow': 3.5,
                         'basin_gwstor_minarea_wb': 3.5,
                         'basin_cfs': 4199.33925198856,
                         'basin_gwflow_cfs': 1387.827901022752,
                         'basin_sroff_cfs': 2270.41420119444,
                         'basin_ssflow_cfs': 4301.081191267585,
                         'runoff_cfs': 7323.828480792126,
                         '14152000': 4199.33925198856,
                         '14150000': 2821.3083223378762,
                         '14148000': 2859.0640001981496,
                         '14145500': 1621.567912886303,
                         '14144800': 1322.9002600573108,
                         '14144900': 185.74238225572472,
                         '14146500': 110.11960621369612,
                         '14147500': 1467.7904311929235,
                         '14151000': 1016.5543048593603,
                         '14150800': 284.04695648793336,
                         '14150300': 802.2032904951465}

        assert csv.data.describe().mean().to_dict() == expected_mean


class TestOutputCSVFileNotFound:
    """Tests for FileNotFoundError handling in OutputCSV."""

    def test_nonexistent_file_raises(self, tmp_path):
        """OutputCSV raises FileNotFoundError for a missing file."""
        missing = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError, match="CSV output file not found"):
            OutputCSV(missing)

    def test_nonexistent_file_str_path(self, tmp_path):
        """OutputCSV raises FileNotFoundError when given a string path."""
        missing = str(tmp_path / "missing.csv")

        with pytest.raises(FileNotFoundError, match="CSV output file not found"):
            OutputCSV(missing)


class TestOutputVariableFileNotFound:
    """Tests for FileNotFoundError handling in OutputVariable."""

    def test_nonexistent_file_raises(self, tmp_path):
        """OutputVariable raises FileNotFoundError for a missing file."""
        missing = tmp_path / "no_such_file.csv"
        metadata = {"test_var": {"dimensions": ["nhru"], "datatype": "float32",
                                 "description": "test", "units": "mm"}}

        with pytest.raises(FileNotFoundError, match="Output variable file not found"):
            OutputVariable("test_var", missing, metadata)

    def test_nonexistent_file_str_path(self, tmp_path):
        """OutputVariable raises FileNotFoundError when given a string path."""
        missing = str(tmp_path / "nope.csv")
        metadata = {"test_var": {"dimensions": ["nhru"], "datatype": "float32",
                                 "description": "test", "units": "mm"}}

        with pytest.raises(FileNotFoundError, match="Output variable file not found"):
            OutputVariable("test_var", missing, metadata)
