"""
Tests for the chimp.data.dem module.
"""
import numpy as np
from pansat.geometry import LonLatRect

from chimp.data.dem import globe


def test_find_files():
    """
    Test finding of NOAA GLOBE files.
    """
    roi = LonLatRect(-106, 39, -104, 41)
    recs = globe.find_files(
        np.datetime64("2000-01-01"),
        np.datetime64("2020-01-01"),
        time_step=np.timedelta64(3, "h"),
        roi=roi
    )
    assert len(recs) == 1
