import pytest
from shutil import copytree
from pathlib import Path


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
