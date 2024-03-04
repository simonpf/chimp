"""
Tests for the chimp.data.input module
=====================================
"""
import numpy as np
import torch

from chimp.data.gpm import GMI
from chimp.data.input import InputLoader, SequenceInputLoader


def test_load_sample(gmi_data):
    """
    Test loading a training sample from CHIMP GMI training files and
    ensure that a torch.Tensor of the correct shape is returned.

    Note: Since the nominal scale of GMI is 4, the returned tensor
    should have width and height 128 if the base scale is 2.
    """
    rng = np.random.default_rng()

    gmi_files = sorted(list((gmi_data / "gmi").glob("*.nc")))
    x = GMI.load_sample(
        gmi_files[0],
        crop_size=(256, 256),
        base_scale=2,
        slices=(10, 266, 20, 276),
        rng=rng,
        rotate=90.0,
        flip=True
    )

    assert isinstance(x, torch.Tensor)
    assert x.shape == (13, 128, 128)


def test_input_loader(gmi_data):
    """
    Test InputLoader for single-step input data.
    """
    input_loader = InputLoader(
        gmi_data,
        input_datasets=["gmi"],
    )

    assert len(input_loader) == 3

    x = input_loader.get_input(np.datetime64("2020-01-01"))

    assert isinstance(x, dict)
    assert "gmi" in x


def test_sequence_input_loader(cpcir_data, gmi_data):
    """
    Test InputLoader for single-step input data.
    """
    input_loader = SequenceInputLoader(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        sequence_length=8
    )

    assert len(input_loader) == 12

    x = input_loader.get_input(np.datetime64("2020-01-01"))
    assert "cpcir" in x
    assert len(x["cpcir"]) == 8
    assert len(x["gmi"]) == 8
