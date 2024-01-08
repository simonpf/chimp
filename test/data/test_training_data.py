"""
Tests for chimp.data.training_data.
==================================
"""
import os
from pathlib import Path

from conftest import mrms_surface_precip_data, cpcir_data, gmi_data
import numpy as np
import pytest
import torch

from chimp.areas import CONUS
from chimp.data.training_data import (
    SingleStepDataset,
    SequenceDataset,
    CHIMPPretrainDataset,
)

TEST_DATA = os.environ.get("CHIMP_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


def test_find_files_single_step(cpcir_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    training_data = SingleStepDataset(
        cpcir_data, reference_data="mrms", inputs=["cpcir"], sample_rate=1
    )
    assert len(training_data) == 24


def test_sparse_data(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are set to None.
    """
    training_data = SingleStepDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_value_policy="sparse",
        window_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)


def test_full_domain(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test loading of full domain data.
    """
    for i in range(10):
        inputs = ["cpcir", "gmi"]
        training_data = CHIMPPretrainDataset(
            cpcir_data,
            reference_data="mrms",
            inputs=inputs,
            missing_value_policy="sparse",
        )

        iter = training_data.full_domain()
        time, x, y = next(iter)

        assert x["cpcir"].shape[-2:] == (960, 1920)


def test_missing_input_policies(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are handled correctly.
    """
    training_data = SingleStepDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_value_policy="sparse",
        window_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_value_policy="random",
        window_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["gmi"].numpy()))
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_value_policy="missing",
        window_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["gmi"].numpy()))
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_value_policy="mean",
        window_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["gmi"].numpy()))
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))


###############################################################################
# Sequence dataset
###############################################################################


def test_find_files_sequence(cpcir_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    training_data = SequenceDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir"],
        sample_rate=1,
        sequence_length=8,
    )
    assert len(training_data) == 16


def test_load_sample_sequence(cpcir_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    training_data = SequenceDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir"],
        sample_rate=1,
        sequence_length=8,
    )
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 8

    # Test reduced output size.
    training_data = SequenceDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir"],
        sample_rate=1,
        sequence_length=8,
        shrink_output=2,
    )
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 8
    assert y["surface_precip"][0].shape[-1] == x["cpcir"][0].shape[-1] // 2
