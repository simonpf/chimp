"""
Tests for the chimp.data.baltrad module.
========================================
"""
import os

import numpy as np
import pytest


from chimp.areas import NORDICS
from chimp.data.baltrad import BALTRAD


BALTRAD_DATA = os.environ.get("BALTRAD_DATA_PATH", None)
HAS_BALTRAD_DATA = BALTRAD_DATA is not None
NEEDS_BALTRAD_DATA = pytest.mark.skipif(not HAS_BALTRAD_DATA, reason="Needs BALTRAD data.")


@NEEDS_BALTRAD_DATA
def test_find_files():
    files = BALTRAD.find_files(
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2019-01-01T00:00:00"),
        np.timedelta64(15, "m"),
        path=BALTRAD_DATA
    )
    assert len(files) == 1


@NEEDS_BALTRAD_DATA
def test_process_file(tmp_path):
    files = BALTRAD.find_files(
        "2019-01-01T00:00:00",
        "2019-01-01T00:00:00",
        np.timedelta64(15, "m"),
        path=BALTRAD_DATA
    )

    BALTRAD.process_file(
        files[0],
        NORDICS,
        tmp_path,
        np.timedelta64(15, "m")
    )

    training_files = BALTRAD.find_training_files(tmp_path)
    assert len(training_files) == 1
    crop_size = NORDICS[4].shape
    ref_data = BALTRAD.load_sample(
        training_files[0],
        crop_size=crop_size,
        base_scale=4,
        slices=(0, crop_size[0], 0, crop_size[1]),
        rng=None
    )
    assert "reflectivity" in ref_data
