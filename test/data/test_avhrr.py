"""
Tests for the chimp.data.avhrr module.
======================================
"""
import os

import numpy as np
import pytest

from chimp.areas import ARCTIC
from chimp.data.avhrr import AVHRR

AVHRR_DATA = os.environ.get("AVHRR_DATA_PATH", None)
HAS_AVHRR_DATA = AVHRR_DATA is not None
NEEDS_AVHRR_DATA = pytest.mark.skipif(not HAS_AVHRR_DATA, reason="Needs AVHRR data.")


@NEEDS_AVHRR_DATA
def test_find_files():
    files = AVHRR.find_files(
        start_time=np.datetime64("2010-01-01"),
        end_time=np.datetime64("2010-01-01"),
        time_step=np.timedelta64(1, "h"),
        path=AVHRR_DATA
    )

@NEEDS_AVHRR_DATA
def test_process_file(tmp_path):
    files = AVHRR.find_files(
        start_time=np.datetime64("2010-01-01"),
        end_time=np.datetime64("2010-01-01"),
        time_step=np.timedelta64(1, "h"),
        path=AVHRR_DATA
    )
    AVHRR.process_file(files[0], domain=ARCTIC, output_folder=tmp_path, time_step=np.timedelta64(1, "h"))
    matches = sorted(list((tmp_path / "avhrr").glob("*.nc")))
    assert len(matches) == 2
