"""
Tests for the chimp.data.input module
=====================================
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from chimp.data.gpm import GMI
from chimp.data.input import (
    InputLoader,
    SequenceInputLoader,
    get_input_map,
    get_input_age
)


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
        base_scale=2, slices=(10, 266, 20, 276),
        rng=rng,
        rotate=90.0,
        flip=True
    )

    assert isinstance(x, torch.Tensor)
    assert x.shape == (13, 128, 128)


@pytest.mark.parametrize("glob", (False, True))
def test_input_loader(gmi_data: Path, glob: bool):
    """
    Test InputLoader for single-step input data.
    """
    input_loader = InputLoader(
        (
            gmi_data if not glob else
            list((gmi_data / "gmi").glob("*.nc"))
        ),
        input_datasets=["gmi"],
    )

    assert len(input_loader) == 3

    x = input_loader.get_input(np.datetime64("2020-01-01"))

    assert isinstance(x, dict)
    assert "gmi" in x


@pytest.mark.parametrize("glob", (False, True))
def test_sequence_input_loader(
    cpcir_data: Path,
    gmi_data: Path,
    glob: bool,
):
    """
    Test InputLoader for single-step input data.
    """
    input_loader = SequenceInputLoader(
        (
            cpcir_data if not glob else (
              list((cpcir_data / "cpcir").glob("*.nc"))
              + list((gmi_data / "gmi").glob("*.nc"))
            )
        ),
        input_datasets=["cpcir", "gmi"],
        sequence_length=8
    )

    assert len(input_loader) == 12

    x = input_loader.get_input(np.datetime64("2020-01-01"))
    assert "cpcir" in x
    assert len(x["cpcir"]) == 8
    assert len(x["gmi"]) == 8


def test_get_input_map():
    """
    Test calculation of the input map.
    """
    x_cpcir = torch.rand(2, 10, 10, 10)
    x_cpcir[x_cpcir < 0.5] = torch.nan
    x_gmi = torch.rand(2, 10, 5, 5)
    x_gmi[x_gmi < 0.5] = torch.nan
    inputs = {
        "cpcir": x_cpcir,
        "gmi": x_gmi
    }
    input_map = get_input_map(inputs)
    assert (input_map[:, 0] == torch.isfinite(x_cpcir).any(1)).all()

    inputs = {
        "cpcir": [x_cpcir, x_cpcir],
        "gmi": [x_gmi, x_gmi]
    }
    input_map = get_input_map(inputs)
    assert isinstance(input_map, list)
    assert (input_map[0][:, 0] == torch.isfinite(x_cpcir).any(1)).all()


def test_get_input_age():
    """
    Test calculation of the input map.
    """
    x_cpcir = []
    for i in range(8):
        x_cpcir.append(torch.rand(2, 1, 10, 10))
        x_cpcir[-1][x_cpcir[-1] > 0.5] = np.nan
    x_gmi = []
    for i in range(8):
        x_gmi.append(torch.rand(2, 1, 10, 10))
        x_gmi[-1][x_gmi[-1] > 0.5] = np.nan
    inputs = {
        "cpcir": x_cpcir,
        "gmi": x_gmi
    }

    ages = get_input_age(inputs)

    for step, age in enumerate(ages):
        if step > 0:
            age = age.to(dtype=torch.int64)
            age_1 = age[:, 0] == 1
            assert age_1.sum() > 0
            x_cpcir = inputs["cpcir"][step - 1]
            assert torch.isfinite(x_cpcir.sum(1)[age_1]).all()
        if step < len(ages) - 1:
            age = age.to(dtype=torch.int64)
            age_1 = age[:, 0] == -1
            assert age_1.sum() > 0
            x_cpcir = inputs["cpcir"][step + 1]
            assert torch.isfinite(x_cpcir.sum(1)[age_1]).all()

    ages = torch.stack(get_input_age(inputs, bidirectional=False))

    assert torch.min(ages[torch.isfinite(ages)]) == 0.0
