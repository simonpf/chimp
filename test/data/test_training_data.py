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
    SingleStepPretrainDataset,
    SequenceDataset,
    expand_times_and_files,
    find_sequence_starts_and_ends
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


def test_single_step_pretrain_dataset(
        cpcir_data,
        gmi_data,
        mrms_surface_precip_data
):
    """
    Instantiate single-step pretrain dataset and ensure that:
        - The identified training samples matches the expected number.

    """
    training_data = SingleStepPretrainDataset(
        cpcir_data,
        input_datasets=["cpcir", "gmi"],
        reference_datasets=["mrms"],
        sample_rate=1
    )
    assert len(training_data) == 24

    x, y = training_data[0]


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

    training_data = SequenceDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=2,
        sequence_length=8,
    )
    assert len(training_data) == 4

    training_data = SequenceDataset(
        cpcir_data,
        input_datasets=["cpcir"],
        reference_datasets=["mrms"],
        sample_rate=0.5,
        sequence_length=8,
    )
    assert len(training_data) == 1

    for x, y in training_data:
        assert len(x["cpcir"]) == 8
        assert len(y["surface_precip"]) == 8
        for y_sp in y["surface_precip"]:
            assert y_sp.isfinite().any()


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
    assert len(training_data) == 3
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


def test_expand_times_and_files():
    """
    Test expansion of times and file arrays.
    """
    times = np.array([
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:15:00"),
        np.datetime64("2020-01-01T01:00:00"),
    ])
    reference_files = np.array([
        ["file_1"],
        ["file_2"],
        ["file_3"],
    ])
    input_files = np.array([
        ["file_1"],
        [None],
        ["file_3"],
    ])

    full = expand_times_and_files(
        times,
        input_files,
        reference_files,
    )
    times_full, input_files_full, reference_files_full = full

    assert times_full.shape[0] == 5
    assert input_files_full.shape[0] == 5
    assert reference_files_full.shape[0] == 5

    assert reference_files_full[0, 0] == "file_1"
    assert reference_files_full[1, 0] == "file_2"
    assert reference_files_full[4, 0] == "file_3"


def test_find_sequence_starts_and_ends():
    """
    Test calculation of sequence starts and ends and ensure that they match
    expected values.
    """
    input_files = np.array([
        ["file_1"],
        [None],
        ["file_2"],
        [None],
        [None],
        ["file_3"],
    ])

    reference_files = input_files

    starts, ends = find_sequence_starts_and_ends(
        input_files,
        reference_files,
        2, 0, True
    )
    assert (starts == [0, 2, 4]).all()

    starts, ends = find_sequence_starts_and_ends(
        input_files,
        reference_files,
        2, 1, True
    )
    assert (starts ==  [0]).all()

    starts, ends = find_sequence_starts_and_ends(
        input_files,
        reference_files,
        2, 1, False
    )
    assert (starts ==  [0]).all()

    # All files are available.
    input_files = np.array([
        ["file_1"],
        ["file_2"],
        ["file_3"],
        ["file_4"],
        ["file_5"],
        ["file_6"],
    ])

    reference_files = input_files
    starts, ends = find_sequence_starts_and_ends(
        input_files,
        reference_files,
        2, 0, True
    )
    assert (starts ==  [0, 2, 4]).all()


def test_load_sparse_data(cpcir_data, cmb_surface_precip_data):
    """
    Test that scene without valid reference data are handled correctly.
    """
    training_data = SequenceDataset(
        cpcir_data,
        reference_datasets=["cmb"],
        input_datasets=["cpcir"],
        sample_rate=1,
        sequence_length=4,
        forecast=4,
        include_input_steps=True
    )
    assert len(training_data) == 2

    for x, y in training_data:
        assert len(x["cpcir"]) == 4
        assert len(y["surface_precip"]) == 8
