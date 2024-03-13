"""
Tests for the chimp.data.gpm module.
"""

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.gpm import GMI, CMB, ATMS_W_ANGLE, DPR


@NEEDS_PANSAT_PASSWORD
def test_find_files_gmi():
    """
    Ensure that GMI files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = GMI.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_gmi(tmp_path):
    """
    Ensure that extraction of observations works as expected.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = GMI.find_files(
        start_time,
        end_time,
        time_step
    )
    GMI.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "gmi").glob("*.nc")))
    assert len(training_files) > 0
    training_data = xr.load_dataset(training_files[0])
    tbs = training_data.tbs.data
    assert np.isfinite(tbs).all(-1).sum() > 100

@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_atms_w_angle(tmp_path):
    """
    Ensure that extraction of ATMS observations works as expected.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = ATMS_W_ANGLE.find_files(
        start_time,
        end_time,
        time_step
    )
    ATMS_W_ANGLE.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "atms_w_angle").glob("*.nc")))
    assert len(training_files) > 0
    training_data = xr.load_dataset(training_files[0])
    tbs = training_data.tbs.data
    assert np.isfinite(tbs).all(-1).sum() > 100
    assert "incidence_angle" in training_data

@NEEDS_PANSAT_PASSWORD
def test_find_files_cmb():
    """
    Ensure that GPM CMB files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = CMB.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_cmb(tmp_path):
    """
    Ensure that extraction of GPM CMB reference data works as expected.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = CMB.find_files(
        start_time,
        end_time,
        time_step
    )
    CMB.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "cmb").glob("*.nc")))
    assert len(training_files) > 0
    training_data = xr.load_dataset(training_files[0])
    sp = training_data.surface_precip.data
    assert np.isfinite(sp).sum() > 100


@NEEDS_PANSAT_PASSWORD
def test_find_files_dpr():
    """
    Ensure that GPM DPR files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = DPR.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_files_dpr(tmp_path):
    """
    Ensure that extraction of GPM DPR reference data works as expected.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = DPR.find_files(
        start_time,
        end_time,
        time_step
    )
    DPR.process_file(
        files[0],
        CONUS_PLUS,
        tmp_path,
        time_step=time_step
    )
    training_files = sorted(list((tmp_path / "dpr").glob("*.nc")))
    assert len(training_files) > 0
    training_data = xr.load_dataset(training_files[0])
    ref = training_data.reflectivity.data
    assert np.isfinite(ref).any((-2, -1)).sum() > 100
