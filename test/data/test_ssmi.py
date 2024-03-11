"""
Tests for the chimp.data.ssmi module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import MERRA
from chimp.data.ssmi import SSMI


@NEEDS_PANSAT_PASSWORD
def test_find_files_ssmi():
    """
    Ensure that SSMI files are found for a given time range.
    """
    start_time = np.datetime64("2023-01-01T00:01:00")
    end_time = np.datetime64("2023-01-01T00:57:00")
    time_step = np.timedelta64(1, "D")
    files = SSMI.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 4

    local_files = SSMI.find_files(
        start_time,
        end_time,
        time_step,
        files[0].parent
    )
    assert len(files) == 4


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_ssmi(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2023-01-01T01:00:00")
    end_time = np.datetime64("2023-01-01T02:00:00")
    time_step = np.timedelta64(1, "D")
    files = SSMI.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 4

    SSMI.process_file(
        files[1],
        MERRA,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "ssmi").glob("*.nc")))
    assert len(training_files) == 1
    training_data = xr.load_dataset(training_files[0])
    n_valid = np.isfinite(training_data.obs.data).sum()
    assert n_valid > 0

    SSMI.process_file(
        files[2],
        MERRA,
        tmp_path,
        time_step=time_step
    )
    training_data = xr.load_dataset(training_files[0])
    n_valid_2 = np.isfinite(training_data.obs.data).sum()
    assert n_valid_2 > n_valid
