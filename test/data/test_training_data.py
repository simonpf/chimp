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
from torch.utils.data import DataLoader
import xarray as xr

from chimp.areas import CONUS
from chimp.data.training_data import (
    SingleStepDataset,
    SequenceDataset,
    CHIMPPretrainDataset,
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

    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=0.5,
        start_time=np.datetime64("2020-01-01T06:00:00"),
        end_time=np.datetime64("2020-01-01T12:00:00")
    )
    assert len(training_data) == 6


def test_load_full_input(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are handled correctly.
    """
    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        scene_size=-1,
    )
    x, y = training_data[1]
    assert x["cpcir"].shape[1:] == (960, 1920)



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
    assert len(training_data) == 2


def test_load_sample_sequence(cpcir_data, mrms_surface_precip_data):
    """
    Instantiate a sequence dataset and ensure that:
        - The expected number of training samples is found.
        - Inputs are lists of length equal to sequence length
        - shrink_output returns outputs that are smaller by the required
          factor.
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
    assert len(training_data) == 1
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 4
    assert "lead_time" in x
    assert len(x["lead_time"]) == 4

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
    assert len(training_data) == 1
    x, y = training_data[0]
    assert len(x["cpcir"]) == 8
    assert len(y["surface_precip"]) == 12
    assert y["surface_precip"][0].shape[-1] == x["cpcir"][0].shape[-1] // 2
    assert "lead_time" in x
    assert len(x["lead_time"]) == 4


def test_load_full_input_sequence(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Ensure that loading of full input scene works.
    """
    training_data = SequenceDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        scene_size=-1,
        sequence_length=8
    )
    x, y = training_data[1]
    assert x["cpcir"][0].shape[1:] == (960, 1920)


def test_load_empty_scenes(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that scene without valid reference data are handled correctly.
    """
    mrms_files = sorted(list((mrms_surface_precip_data / "mrms").glob("*.nc")))
    for mrms_file in mrms_files[1:]:
        mrms_data = xr.load_dataset(mrms_file)
        mrms_data["rqi"].data[:] = 0.0
        mrms_data.to_netcdf(mrms_file)

    training_data = SingleStepDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        scene_size=256,
        validation=True
    )

    data_loader = DataLoader(training_data, batch_size=8)
    x, y = next(iter(data_loader))
    assert "cpcir" in x
