from pathlib import Path
import os

import numpy as np
import pytest
import xarray as xr

from cimr.data.cpcir import resample_data
from cimr.areas import CONUS

TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

@NEEDS_TEST_DATA
def test_resample_data():
    """
    Ensure that resample CPCIR data to a given domain yields the
    expected results.
    """
    cpcir_file = TEST_DATA / "cpcir" / "merg_2020010100_4km-pixel.nc4"
    cpcir_data = xr.load_dataset(cpcir_file)
    tbs_r = resample_data(CONUS, cpcir_data[{"time": 0}])
    assert np.any(np.isfinite(tbs_r.tbs.data))
    assert tbs_r.tbs.shape[1:] == CONUS[4].shape
