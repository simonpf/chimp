import numpy as np
import os
from pathlib import Path

import pytest
import xarray as xr

from cimr.areas import CONUS_4
from cimr.data.resample import resample_swath_center, resample_tbs
from cimr.areas import NORDICS
from cimr.data.gmi import (
    l1c_gpm_gmi_r,
)

data_path = Path(__file__).parent / "data"
gmi_file = "1C-R.GPM.GMI.XCAL2016-C.20200501-S075828-E093100.035075.V07A.HDF5"


TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


def test_resample_swath_center():
    """
    Ensure that resampling the longitude and latitude coordinates of a
    rectangular domain returns a vertical line at the center of the domain.
    """
    lons, lats = CONUS_4.get_lonlats()
    row_indices, col_indices = resample_swath_center(
        CONUS_4,
        lons,
        lats,
        radius_of_influence=10e3
    )
    assert np.all(col_indices == lats.shape[1] // 2)

    # Ensure that swath outside domain returns empty arrays.
    lons[:] = 0.0
    row_indices, col_indices = resample_swath_center(
        CONUS_4,
        lons,
        lats,
        radius_of_influence=10e3
    )
    assert len(row_indices) == 0


@NEEDS_TEST_DATA
def test_resample_gmi_tbs():
    """
    Test resampling of GMI TBS.
    """
    domain = NORDICS
    product = l1c_gpm_gmi_r
    data = product.open(data_path / "obs" / gmi_file)
    tbs_r = resample_tbs(
        domain[8],
        data,
        2,
        radius_of_influence=15e3,
        include_scan_time=True
    )

    assert tbs_r.channels.size == 13
    assert np.any(np.isfinite(tbs_r["tbs"]))
    assert tbs_r.swath_centers.size > 0
    assert "scan_time" in tbs_r
