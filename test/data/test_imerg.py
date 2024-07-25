"""
Tests for the IMERG reference data provided by chimp.data.imerg
"""
from conftest import NEEDS_PANSAT_PASSWORD

import pytest
import numpy as np
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.imerg import (
    IMERG_EARLY,
    IMERG_LATE,
    IMERG_FINAL
)

@NEEDS_PANSAT_PASSWORD
@pytest.mark.parametrize("dataset", [IMERG_EARLY, IMERG_LATE, IMERG_FINAL])
def test_find_files_imerg(dataset):
    """
    Ensure that IMERG files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = dataset.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
@pytest.mark.parametrize("dataset", [IMERG_EARLY, IMERG_LATE, IMERG_FINAL])
def test_process_files(dataset, tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = dataset.find_files(
        start_time,
        end_time,
        time_step
    )
    dataset.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / dataset.name).glob("*.nc")))
    assert len(training_files) > 0
    training_data = xr.load_dataset(training_files[0])
    surface_precip = training_data.surface_precip.data
    assert np.isfinite(surface_precip).all(-1).sum() > 100
