import pytest
# import numpy as np
from pyPRMS import MetaData


class TestMetaData:

    def test_metadata_version_ref6(self):
        """Test reading metadata when version == 60.0

        PRMS-classic released a version 6.0.0 in 2025-02 which conflicts
        with the earlier prms6 refactored development code. Metadata version information
        for the earlier development code was changed from 6.0 to 60.0 to avoid conflicts.
        """
        prms_meta = MetaData(version='60.0', verbose=True).metadata
        assert prms_meta['control'].get('stat_var_file', None) is None

    def test_metadata_version6(self):
        prms_meta = MetaData(version='6.0.0', verbose=True).metadata
        assert prms_meta['parameters'].get('outVarON_OFF', None) is None

    def test_metadata_version5(self):
        prms_meta = MetaData(version='5.2.1.1', verbose=True).metadata
        assert prms_meta['parameters'].get('outVarON_OFF', None) is None

    def test_metadata_version4(self):
        prms_meta = MetaData(version='4', verbose=True).metadata
        assert prms_meta['parameters'].get('soilzone_aet_flag', None) is None
