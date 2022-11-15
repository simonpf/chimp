"""
Tests for the cimr.data.gmi module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from cimr.areas import NORDIC
from cimr.data.gmi import (
    l1c_gpm_gmi_r,
    resample_swaths,
    process_file
)
data_path = Path(__file__).parent / "data"
gmi_file = "1C-R.GPM.GMI.XCAL2016-C.20200501-S075828-E093100.035075.V07A.HDF5"


def test_resampling():
    """
    Ensure that resampling produces valid brightness temperatures.
    """
    domain = NORDIC
    product = l1c_gpm_gmi_r
    data = product.open(data_path / "obs" / gmi_file)
    tbs_r = resample_swaths(domain, data)

    assert "tbs_s1" in tbs_r
    assert "tbs_s2" in tbs_r

    assert np.any(np.isfinite(tbs_r["tbs_s1"]))


def test_process_file(tmp_path):
    """
    Enusre that processing a single file produces a training data file
    with the expected input.
    """
    domain = NORDIC
    product = l1c_gpm_gmi_r
    data = product.open(data_path / "obs" / gmi_file)
    process_file(domain, data, tmp_path)

    files = list(tmp_path.glob("*.nc"))
    assert len(files) == 1

    data = xr.load_dataset(files[0])
    mw_low = data["mw_low"]
    assert mw_low.shape[-1] == 7
    assert np.any(np.isfinite(mw_low[..., 0]))
    assert np.any(np.isfinite(mw_low[..., 1]))
    assert np.any(np.isfinite(mw_low[..., 2]))
    assert np.any(np.isfinite(mw_low[..., 3]))
    assert np.any(np.isfinite(mw_low[..., 4]))
    assert np.any(np.isfinite(mw_low[..., 5]))
    assert np.any(np.isfinite(mw_low[..., 6]))

    mw_90 = data["mw_90"]
    assert mw_90.shape[-1] == 2
    assert np.any(np.isfinite(mw_90[..., 0]))
    assert np.any(np.isfinite(mw_90[..., 1]))

    mw_160 = data["mw_160"]
    assert mw_160.shape[-1] == 2
    assert np.any(np.isfinite(mw_160[..., 0]))
    assert np.any(np.isfinite(mw_160[..., 1]))

    mw_183 = data["mw_183"]
    assert mw_183.shape[-1] == 5
    assert np.any(np.isnan(mw_183[..., 0]))
    assert np.all(np.isnan(mw_183[..., 1]))
    assert np.any(np.isfinite(mw_183[..., 2]))
    assert np.all(np.isnan(mw_183[..., 3]))
    assert np.any(np.isfinite(mw_183[..., 4]))
