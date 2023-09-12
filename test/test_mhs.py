"""
Tests for the cimr.data.mhs module.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from cimr.areas import NORDICS
from cimr.data.mhs import (
    MHS_PRODUCTS,
    resample_swaths,
    process_file
)

data_path = Path(__file__).parent / "data"
mhs_file = "1C.METOPB.MHS.XCAL2016-V.20200501-S085148-E103309.039533.V05A.HDF5"


def test_resampling():
    """
    Ensure that resampling produces valid brightness temperatures.
    """
    domain = NORDICS
    product = MHS_PRODUCTS[0]
    data = product.open(data_path / "obs" / mhs_file)
    tbs_r = resample_swaths(domain, data)

    assert np.any(np.isfinite(tbs_r))


def test_process_file(tmp_path):
    """
    Enusre that processing a single file produces a training data file
    with the expected input.
    """
    domain = NORDICS
    product = MHS_PRODUCTS[0]
    data = product.open(data_path / "obs" / mhs_file)
    process_file(domain, data, tmp_path)

    files = list(tmp_path.glob("*.nc"))
    assert len(files) == 1

    data = xr.load_dataset(files[0])

    mw_90 = data["mw_90"]
    assert mw_90.shape[-1] == 2
    assert np.all(np.isnan(mw_90[..., 0]))
    assert np.any(np.isfinite(mw_90[..., 1]))

    mw_160 = data["mw_160"]
    assert mw_160.shape[-1] == 2
    assert np.all(np.isnan(mw_160[..., 0]))
    assert np.any(np.isfinite(mw_160[..., 1]))

    mw_183 = data["mw_183"]
    assert mw_183.shape[-1] == 5
    assert np.any(np.isfinite(mw_183[..., 0]))
    assert np.all(np.isnan(mw_183[..., 1]))
    assert np.any(np.isfinite(mw_183[..., 2]))
    assert np.all(np.isnan(mw_183[..., 3]))
    assert np.any(np.isfinite(mw_183[..., 4]))
