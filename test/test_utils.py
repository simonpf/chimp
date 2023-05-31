""""
Test for the cimr.utils module.
"""
from datetime import datetime

from cimr.utils import round_time


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
