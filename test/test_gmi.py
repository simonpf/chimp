"""
Tests for the cimr.data.gmi module.
"""
from pathlib import Path
from datetime import timedelta

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


def test_process_file(tmp_path):
    """
    Ensure that processing a single file produces a training data file
    with the expected input.
    """
    domain = NORDIC
    product = l1c_gpm_gmi_r
    data = product.open(data_path / "obs" / gmi_file)
    process_file(domain, data, tmp_path, timedelta(minutes=15))

    files = list(tmp_path.glob("*.nc"))
    assert len(files) == 1

    data = xr.load_dataset(files[0])

    tbs = data["tbs"]
    assert tbs.shape[0] == 13
    assert NORDIC[8].shape == tbs.shape[1:]
    for i in range(13):
        assert np.any(tbs.data[i])
