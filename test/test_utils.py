""""
Test for the cimr.utils module.
"""
from datetime import datetime
from pathlib import Path
import os

import pytest

from cimr.utils import round_time, get_available_times, get_date


TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

def test_round_time():
    """
    Assert that rounding to different time intervals works as expected.
    """
    time = datetime(2020, 1, 1, 0, 13)
    time_r = round_time(time, minutes=15)
    assert time_r.minute == 15
    time = datetime(2020, 1, 1, 0, 17)
    time_r = round_time(time, minutes=15)
    assert time_r.minute == 15

    time = datetime(2020, 1, 1, 0, 13)
    time_r = round_time(time, minutes=30)
    assert time_r.minute == 0
    time = datetime(2020, 1, 1, 0, 17)
    time_r = round_time(time, minutes=30)
    assert time_r.minute == 30


def test_get_date():
    """
    Ensure that extracting the date from a CIMR filename works.
    """
    filename = "mhs_20200101_00_00.nc"
    date = get_date(filename)

    assert date == datetime(2020, 1, 1, 0, 0)


@NEEDS_TEST_DATA
def test_get_available_times():
    """
    Assert that rounding to different time intervals works as expected.
    """
    times = get_available_times(TEST_DATA / "training_data" / "cpcir")
    assert len(times) == 4
    assert times[0] == datetime(2020, 1, 1)
