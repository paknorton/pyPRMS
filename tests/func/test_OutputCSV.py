import pytest

from pyPRMS.summary.OutputCSV import OutputCSV
from pyPRMS.summary.OutputVariable import OutputVariable


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
