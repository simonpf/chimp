import numpy as np

from chimp.data.utils import (
    scale_slices,
    round_time
)



def test_scale_slices():
    """
    Test scaling of slices works as expected.
    """
    slcs = scale_slices(None, 2)
    assert slcs is not None

    slcs = (slice(0, 2), slice(0, 2))
    slcs_scld = scale_slices(slcs, 2)
    assert slcs_scld[0].start == 0
    assert slcs_scld[0].stop == 1
    assert slcs_scld[1].start == 0
    assert slcs_scld[1].stop == 1

    slcs = (slice(0, 2), slice(0, 2))
    slcs_scld = scale_slices(slcs, 0.5)
    assert slcs_scld[0].start == 0
    assert slcs_scld[0].stop == 4
    assert slcs_scld[1].start == 0
    assert slcs_scld[1].stop == 4


def test_round_time():
    """
    Test the rounding of times.
    """
    ref_time = np.datetime64("2020-01-01T01:15:00", "s")

    datetime = np.datetime64("2020-01-01T01:20:00", "s")
    step = np.timedelta64(15, "m")
    rounded = round_time(datetime, step)
    assert rounded == ref_time

    datetime = np.datetime64("2020-01-01T01:25:00", "s")
    rounded = round_time(datetime, step)
    assert rounded == ref_time
