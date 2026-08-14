import pytest
from shutil import copytree
from pathlib import Path

def pytest_collection_modifyitems(items):
    """Modifies test items in place to ensure test classes run in a given order."""
    CLASS_ORDER = ['TestPrmsHelpers',
                   'TestMetaData',
                   'TestDimension', 'TestEmptyDimensions', 'TestEmptyParamDimensions',
                   'TestControlVariable', 'TestControl', 'TestControlFile',
                   'TestParameter', 'TestParameters', 'TestParameterFile', 'TestParamDb', 'TestParameterNetCDF',
                   'TestOutputVariables', 'TestOutputCSV', 'TestOutputCSVFileNotFound', 'TestOutputVariableFileNotFound',
                   'TestDataFile',
                   'TestCbh']
    sorted_items = items.copy()

    # read the class names from default items
    class_mapping = {item: item.cls.__name__ for item in items}

    # Iteratively move tests of each class to the end of the test queue
    for class_ in CLASS_ORDER:
        sorted_items = ([it for it in sorted_items if class_mapping[it] != class_] +
                        [it for it in sorted_items if class_mapping[it] == class_])


    items[:] = sorted_items

@pytest.fixture
def datadir(tmp_path, request, scope='function'):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    FIXTURE_DIR = Path(request.module.__file__).parent.resolve() / Path(request.module.__file__).stem

    if FIXTURE_DIR.is_dir():
        copytree(FIXTURE_DIR, str(tmp_path), dirs_exist_ok=True)

    return tmp_path
