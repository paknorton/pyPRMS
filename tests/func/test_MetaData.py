import pytest
# import numpy as np
from pyPRMS import MetaData


class TestMetaData:

    def test_metadata_version(self):
        """Test reading metadata when version > 5"""
        prms_meta = MetaData(version=6, verbose=True).metadata
        assert prms_meta['control'].get('stat_var_file', None) is None

    def test_metadata_version5(self):
        prms_meta = MetaData(version=5, verbose=True).metadata
        assert prms_meta['parameters'].get('outVarON_OFF', None) is None

    def test_metadata_version4(self):
        prms_meta = MetaData(version=4, verbose=True).metadata
        assert prms_meta['parameters'].get('soilzone_aet_flag', None) is None
