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
    Instantiate single-step dataset and ensure that:
        - The identified training samples matches the expected number.
        - Sub-sampling by time reduces the number of samples.
        - Setting time limits reduces the number of training samples.

    """
    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=1
    )
    assert len(training_data) == 24

    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=1,
        time_step=np.timedelta64(60, "m")
    )
    assert len(training_data) == 12

    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=1,
        start_time=np.datetime64("2020-01-01T06:00:00"),
        end_time=np.datetime64("2020-01-01T12:00:00")
    )
    assert len(training_data) == 12


def test_sparse_data(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are set to None.
    """
    training_data = SingleStepDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir", "gmi"],
        missing_value_policy="sparse",
        scene_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)




def test_missing_input_policies(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are handled correctly.
    """
    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        missing_value_policy="sparse",
        scene_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        missing_value_policy="random",
        scene_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["gmi"].numpy()))
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir", "gmi"],
        missing_value_policy="missing",
        scene_size=128,
    )
    x, y = training_data[1]
    assert x["gmi"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["gmi"].numpy()))
    assert x["cpcir"].shape[1:] == (128, 128)
    assert np.all(np.isfinite(x["cpcir"].numpy()))

    training_data = SingleStepDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir", "gmi"],
        missing_value_policy="mean",
        scene_size=128,
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
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=1,
        sequence_length=8,
    )
    assert len(training_data) == 16


def test_load_sample_sequence(cpcir_data, mrms_surface_precip_data):
    """
    Instantiate a sequence dataset and ensure that:
        - The expected number of training samples is found.
        - Inputs are lists of the sequence length
        - shrink_output:
    """
    training_data = SequenceDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir"],
        sample_rate=1,
        sequence_length=8,
    )
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 8

    # Test reduced output size.
    training_data = SequenceDataset(cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir"],
        sample_rate=1,
        sequence_length=8,
        shrink_output=2,
    )
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 8
    assert y["surface_precip"][0].shape[-1] == x["cpcir"][0].shape[-1] // 2


def test_load_sample_forecast(cpcir_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    training_data = SequenceDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir"],
        sample_rate=1,
        sequence_length=8,
        include_input_steps=False,
        forecast=4
    )
    assert len(training_data) == 12
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 4
    assert "lead_times" in x
    assert len(x["lead_times"]) == 4

    # Test reduced output size.
    training_data = SequenceDataset(
        cpcir_data,
        reference_datasets=["mrms"],
        input_datasets=["cpcir"],
        sample_rate=1,
        sequence_length=8,
        shrink_output=2,
        forecast=4,
        include_input_steps=True
    )
    assert len(training_data) == 12
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 12
    assert y["surface_precip"][0].shape[-1] == x["cpcir"][0].shape[-1] // 2
    assert "lead_times" in x
    assert len(x["lead_times"]) == 4
