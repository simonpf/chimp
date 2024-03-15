"""
Tests for the chimp.data.gpm module.
"""
from pathlib import Path

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest
import xarray as xr

from chimp.areas import CONUS_PLUS
from chimp.data.gpm import GMI, CMB, ATMS_W_ANGLE, DPR
from chimp.data.utils import get_output_filename
from chimp.data.training_data import SingleStepDataset


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



def write_cmb_file(path: Path, time: np.datetime64, time_step: np.timedelta64) -> None:
    """
    Write dummy CMB file filed with surface precip values of 2 across the right half
    of the domain.
    """
    data = xr.Dataset({
        "surface_precip": (("y", "x"), 2.0 * np.ones((512, 512), dtype=np.float32))
    })
    data.surface_precip.data[:, :256] = np.nan
    filename = get_output_filename("cmb", time, time_step)
    data.to_netcdf(path / "cmb" / filename)


def write_mrms_file(path: Path, time: np.datetime64, time_step: np.timedelta64) -> None:
    """
    Write dummy MRMS file filed with surface precip values of 2 across the right half
    of the domain.
    """
    data = xr.Dataset({
        "surface_precip": (("y", "x"), 1.0 * np.ones((512, 512), dtype=np.float32)),
        "rqi": (("y", "x"), 1.0 * np.ones((512, 512), dtype=np.float32))
    })
    filename = get_output_filename("mrms", time, time_step)
    data.to_netcdf(path / "mrms" / filename)


def write_cpcir_file(path: Path, time: np.datetime64, time_step: np.timedelta64) -> None:
    """
    Write dummy MRMS file filed with surface precip values of 2 across the right half
    of the domain.
    """
    data = xr.Dataset({
        "tbs": (("y", "x"), 1.0 * np.ones((512, 512), dtype=np.float32))
    })
    filename = get_output_filename("cpcir", time, time_step)
    data.to_netcdf(path/ "cpcir" / filename)


@pytest.fixture()
def cmb_and_mrms_data(tmp_path):

    time_step = np.timedelta64(15, "m")

    cmb_dir = tmp_path / "cmb"
    cmb_dir.mkdir()
    write_cmb_file(tmp_path, np.datetime64("2023-01-01T00:00:00"), time_step)
    write_cmb_file(tmp_path, np.datetime64("2023-01-01T00:30:00"), time_step)

    mrms_dir = tmp_path / "mrms"
    mrms_dir.mkdir()
    write_mrms_file(tmp_path, np.datetime64("2023-01-01T00:00:00"), time_step)
    write_mrms_file(tmp_path, np.datetime64("2023-01-01T00:15:00"), time_step)
    write_mrms_file(tmp_path, np.datetime64("2023-01-01T00:30:00"), time_step)
    write_mrms_file(tmp_path, np.datetime64("2023-01-01T00:45:00"), time_step)

    cpcir_dir = tmp_path / "cpcir"
    cpcir_dir.mkdir()
    write_cpcir_file(tmp_path, np.datetime64("2023-01-01T00:00:00"), time_step)
    write_cpcir_file(tmp_path, np.datetime64("2023-01-01T00:15:00"), time_step)
    write_cpcir_file(tmp_path, np.datetime64("2023-01-01T00:30:00"), time_step)
    write_cpcir_file(tmp_path, np.datetime64("2023-01-01T00:45:00"), time_step)

    return tmp_path


def test_cmb_and(cmb_and_mrms_data):

    training_data_cmb = SingleStepDataset(
        cmb_and_mrms_data,
        input_datasets=["cpcir"],
        reference_datasets=["cmb"],
        scene_size=-1,
    )
    training_data_cmb_and = SingleStepDataset(
        cmb_and_mrms_data,
        input_datasets=["cpcir"],
        reference_datasets=["cmb_and_mrms"],
        scene_size=-1,
    )

    assert len(training_data_cmb) == 2
    assert len(training_data_cmb_and) == 4

    x_cmb, y_cmb = training_data_cmb[0]
    x_cmb_and, y_cmb_and = training_data_cmb_and[0]

    sp_cmb = y_cmb["surface_precip"]
    sp_cmb_and = y_cmb_and["surface_precip"]

    assert list(np.unique(sp_cmb[np.isfinite(sp_cmb)])) == [2.0]
    assert list(np.unique(sp_cmb_and[np.isfinite(sp_cmb_and)])) == [1.0, 2.0]
