"""
Tests for the chimp.data.opera module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import NORDICS
from chimp.data.opera import (
    OPERA_REFLECTIVITY,
    OPERA_W_PRECIP
)


@NEEDS_PANSAT_PASSWORD
def test_find_files_opera():
    """
    Ensure that OPERA files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:01:00")
    end_time = np.datetime64("2020-01-01T00:57:00")
    time_step = np.timedelta64(15, "m")
    files = OPERA_REFLECTIVITY.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 1

    local_files = OPERA_REFLECTIVITY.find_files(
        start_time,
        end_time,
        time_step,
        files[0].parent
    )
    assert len(files) == 1



@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_opera(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2020-01-01T01:00:00")
    end_time = np.datetime64("2020-01-01T02:00:00")
    time_step = np.timedelta64(1, "h")
    files = OPERA_REFLECTIVITY.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 1
    OPERA_REFLECTIVITY.process_file(
        files[0],
        NORDICS,
        tmp_path,
        time_step=time_step
    )

    training_files = sorted(list((tmp_path / "opera_reflectivity").glob("*.nc")))
    assert len(training_files) == 24
    training_data = xr.load_dataset(training_files[0])
    assert np.isfinite(training_data.reflectivity.data).sum() > 100
    assert np.isfinite(training_data.qi.data).sum() > 100


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_opera_w_precip(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2020-01-01T01:00:00")
    end_time = np.datetime64("2020-01-01T02:00:00")
    time_step = np.timedelta64(1, "h")
    files = OPERA_W_PRECIP.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) == 1
    OPERA_W_PRECIP.process_file(
        files[0],
        NORDICS,
        tmp_path,
        time_step=time_step
    )

    training_files = sorted(list((tmp_path / "opera_reflectivity").glob("*.nc")))
    assert len(training_files) == 24

    _, training_files = OPERA_W_PRECIP.find_training_files(tmp_path)
    assert len(training_files) == 24
    ref_data = OPERA_W_PRECIP.load_sample(
        training_files[0],
        crop_size=256,
        base_scale=4,
        slices=(0, crop_size[0], 0, crop_size[1]),
        rng=None
    )
    assert "reflectivity" in ref_data
    assert "surface_precip_zr" in ref_data
