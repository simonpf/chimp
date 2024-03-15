"""
Tests for the chimp.data.goes module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.goes import GOES_18
from chimp.data.utils import records_to_paths


def test_find_files_goes():
    """
    Ensure that GOES files are found for a given time range.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:05:00")
    time_step = np.timedelta64(15, "m")
    files = GOES_18.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 1
    files = records_to_paths(files)

    local_files = GOES_18.find_files(
        start_time,
        end_time,
        time_step,
        files[0][0].parent
    )
    assert len(files) == 1



@pytest.mark.slow
def test_process_files_goes(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:05:00")
    time_step = np.timedelta64(15, "m")
    files = GOES_18.find_files(
        start_time,
        end_time,
        time_step
    )
    for path in files:
        GOES_18.process_file(
            path,
            CONUS_PLUS,
            tmp_path,
            time_step=time_step
        )
    training_files = sorted(list((tmp_path / "goes_18").glob("*.nc")))
    assert len(training_files) == 1

    training_data = xr.load_dataset(training_files[0])
    for ind in range(6):
        assert np.isfinite(training_data.refls.data[..., ind]).sum() > 0
    for ind in range(10):
        assert np.isfinite(training_data.tbs.data[..., ind]).sum() > 0
