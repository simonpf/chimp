"""
Tests for cimr.data.reference
==================================
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cimr.data.reference import find_random_scene

TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


@NEEDS_TEST_DATA
def test_find_random_scene():

    input_file = TEST_DATA / "training_data" / "radar" / "radar_20200501_08_30.nc"
    rng = np.random.default_rng()

    slices = find_random_scene(
        input_file,
        rng,
        multiple=4,
        window_size=384,
        rqi_thresh=0.8
    )

    assert slices[0] % 4 == 0
    assert slices[1] % 4 == 0
    assert slices[2] % 4 == 0
    assert slices[3] % 4 == 0

    assert slices[1] - slices[0] == 384
