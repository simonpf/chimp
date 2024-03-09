"""
Tests for the chimp.data.gridsat module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.gridsat import GRIDSAT


def test_find_files_gridsat():
    """
    Ensure that GridSat files are found for a given time range.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T22:00:00")
    time_step = np.timedelta64(15, "m")
    files = GRIDSAT.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 8

    local_files = GRIDSAT.find_files(
        start_time,
        end_time,
        time_step,
        files[0].parent
    )
    assert len(files) == 8



@pytest.mark.slow
def test_process_files_gridsat(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T22:00:00")
    time_step = np.timedelta64(15, "m")
    files = GRIDSAT.find_files(
        start_time,
        end_time,
        time_step
    )
    GRIDSAT.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "gridsat").glob("*.nc")))
    assert len(training_files) == 1
    training_data = xr.load_dataset(training_files[0])
    assert training_data.obs.data.shape[2] == 1
    assert np.isfinite(training_data.obs.data).sum() > 100

    [path.unlink() for path in training_files]

    time_step = np.timedelta64(1, "D")
    for path in files:
        GRIDSAT.process_file(
            path,
            CONUS_PLUS,
            tmp_path,
            time_step=time_step
        )
    training_files = sorted(list((tmp_path / "gridsat").glob("*.nc")))
    assert len(training_files) == 1
    training_data = xr.load_dataset(training_files[0])
    assert training_data.obs.data.shape[2] == 8
    for t_ind in range(8):
        assert np.isfinite(training_data.obs.data[:, :, t_ind]).sum() > 100
