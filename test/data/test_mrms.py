"""
Tests for the chimp.data.mrms module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.mrms import MRMS_PRECIP_RATE


def test_find_files_mrms():
    """
    Ensure that MRMS files are found for a given time range.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:57:00")
    time_step = np.timedelta64(15, "m")
    files = MRMS_PRECIP_RATE.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 12

    local_files = MRMS_PRECIP_RATE.find_files(
        start_time,
        end_time,
        time_step,
        files[0].parent
    )
    assert len(files) == 12



@pytest.mark.slow
def test_process_files_mrms(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2023-01-01T00:00:00")
    end_time = np.datetime64("2023-01-01T00:00:00")
    time_step = np.timedelta64(15, "m")
    files = MRMS_PRECIP_RATE.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 3
    for path in files:
        MRMS_PRECIP_RATE.process_file(
            path,
            CONUS_PLUS,
            tmp_path,
            time_step=time_step
        )

    training_files = sorted(list((tmp_path / "mrms").glob("*.nc")))
    assert len(training_files) == 1
    training_data = xr.load_dataset(training_files[0])
    assert np.isfinite(training_data.surface_precip.data).sum() > 100
    assert np.isfinite(training_data.rqi.data).sum() > 100
    assert np.isfinite(training_data.precip_type.data).sum() > 100
