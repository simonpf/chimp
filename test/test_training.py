"""
Test the training of CHIMP retrieval models.
"""
from pathlib import Path
import os

import numpy as np
import pytest


from chimp.training import find_most_recent_checkpoint


def test_find_most_recent_checkpoint(tmp_path):
    """
    Test that the find_most_recent_checkpoint function identifies the
    correct checkpoint file.
    """
    ckpt_1 = tmp_path / ("model.ckpt")
    ckpt_2 = tmp_path / ("model-v1.ckpt")
    ckpt_3 = tmp_path / ("model-v12.ckpt")
    model_name = "model"

    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt is None

    ckpt_1.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_1

    ckpt_2.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_2

    ckpt_3.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_3


