"""
Tests for the chimp.data.wxfm module.
"""

import numpy as np

from conftest import NEEDS_PANSAT_PASSWORD

from chimp.data.wxfm import WXFM_DYNAMIC, WXFM_STATIC
from chimp.areas import MERRA


@NEEDS_PANSAT_PASSWORD
def test_find_files_wxfm_dynamic():
    """
    Ensure that GMI files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = WXFM_DYNAMIC.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0
    assert len(files[0]) == len(WXFM_DYNAMIC.products)


@NEEDS_PANSAT_PASSWORD
def test_process_file_wxfm_dynamic(tmp_path):
    """
    Ensure that GMI files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(180, "m")
    files = WXFM_DYNAMIC.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0
    assert len(files[0]) == len(WXFM_DYNAMIC.products)

    WXFM_DYNAMIC.process_file(
        files[0],
        MERRA,
        tmp_path,
        time_step=time_step
    )
    files = sorted(list((tmp_path / "wxfm_dynamic").glob("*.nc")))
    assert len(files) > 0


@NEEDS_PANSAT_PASSWORD
def test_find_files_wxfm_static():
    """
    Ensure that GMI files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(15, "m")
    files = WXFM_STATIC.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0
    assert len(files[0]) == len(WXFM_STATIC.products)


@NEEDS_PANSAT_PASSWORD
def test_process_file_wxfm_static(tmp_path):
    """
    Ensure that GMI files are found for a given time range.
    """
    start_time = np.datetime64("2020-01-01T00:00:00")
    end_time = np.datetime64("2020-01-01T03:00:00")
    time_step = np.timedelta64(180, "m")
    files = WXFM_STATIC.find_files(
        start_time,
        end_time,
        time_step
    )
    assert len(files) > 0
    assert len(files[0]) == len(WXFM_STATIC.products)

    WXFM_STATIC.process_file(
        files[0],
        MERRA,
        tmp_path,
        time_step=time_step
    )
    files = sorted(list((tmp_path / "wxfm_static").glob("*.nc")))
    assert len(files) > 0
