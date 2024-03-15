"""
Tests for the chimp.data.seviri module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.seviri import SEVIRI


@NEEDS_PANSAT_PASSWORD
def test_find_files_seviri():
    """
    Ensure that SEVIRI files are found for a given time range.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:05:00")
    time_step = np.timedelta64(15, "m")
    files = SEVIRI.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 1

    local_files = SEVIRI.find_files(
        start_time,
        end_time,
        time_step,
        files[0].parent
    )
    assert len(files) == 1



@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_seviri(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:05:00")
    time_step = np.timedelta64(15, "m")
    files = SEVIRI.find_files(
        start_time,
        end_time,
        time_step
    )
    for path in files:
        SEVIRI.process_file(
            path,
            CONUS_PLUS,
            tmp_path,
            time_step=time_step
        )

    training_files = sorted(list((tmp_path / "seviri").glob("*.nc")))
    assert len(training_files) == 1

    training_data = xr.load_dataset(training_files[0])
    for ind in range(12):
        assert np.isfinite(training_data.obs.data[..., ind]).sum() > 0
