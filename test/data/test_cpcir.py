"""
Tests for the chimp.data.cpcir module.
========================================
"""
import os

from conftest import NEEDS_PANSAT_PASSWORD

import numpy as np
import pytest

from chimp.areas import NORDICS
from chimp.data.cpcir import CPCIR


@NEEDS_PANSAT_PASSWORD
def test_find_files():
    files = CPCIR.find_files(
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2019-01-01T00:00:00"),
        np.timedelta64(15, "m"),
    )
    assert len(files) == 1


@pytest.mark.slow
@NEEDS_PANSAT_PASSWORD
def test_process_file(tmp_path):
    files = CPCIR.find_files(
        "2019-01-01T00:00:00",
        "2019-01-01T00:00:00",
        np.timedelta64(15, "m"),
    )

    CPCIR.process_file(
        files[0],
        NORDICS,
        tmp_path,
        np.timedelta64(15, "m")
    )
