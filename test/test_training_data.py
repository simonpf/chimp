"""
Tests for cimr.data.training_data.
==================================
"""
import os
from pathlib import Path

from conftest import (
    mrms_surface_precip_data,
    cpcir_data,
    gmi_data
)
import numpy as np
import pytest
import torch

from cimr.areas import CONUS
from cimr.data.training_data import (
    collate_recursive,
    sparse_collate,
    CIMRDataset,
    CIMRPretrainDataset
)

TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


def test_collate_recursive():
    """
    Test recursive part of collate function.
    """
    x = torch.ones((1))
    y = 2.0 * torch.ones((1))
    c = collate_recursive((x, y))
    assert c == ([x], [y])
    c = collate_recursive((y, x), c)
    assert c == ([x, y], [y, x])

    c = collate_recursive(({"x_geo": x}, y))
    assert c == ({"x_geo": [x]}, [y])
    c = collate_recursive(({"x_geo": y}, x), c)
    assert c == ({"x_geo": [x, y]}, [y, x])

    # Test conversion of numpy types
    x = np.ones((1))
    x_t = torch.as_tensor(x)
    y = 2.0 * np.ones((1))
    y_t = torch.as_tensor(y)
    c = collate_recursive((x, y))
    c = collate_recursive((y, x), c)

    assert (c[0][0] == 1.0).all()
    assert (c[0][1] == 2.0).all()
    assert (c[1][0] == 2.0).all()
    assert (c[1][1] == 1.0).all()


def test_sparse_collate():
    """
    Test collate function for sparse data.
    """
    x = torch.ones((1))
    y = 2.0 * torch.ones((1))
    b = [(x, y), (None, y), (x, None), (None, None)]
    x, y = sparse_collate(b)

    assert x.batch_size == 4
    assert x.batch_indices == [0, 2]
    assert y.batch_size == 4
    assert y.batch_indices == [0, 1]


    x = torch.ones((1))
    y = 2.0 * torch.ones((1))
    b = [
        ({"geo": x}, y),
        ({"geo": None}, y),
        ({"geo": x}, None),
        ({"geo": None}, None)
    ]
    x, y = sparse_collate(b)

    assert x["geo"].batch_size == 4
    assert x["geo"].batch_indices == [0, 2]
    assert y.batch_size == 4
    assert y.batch_indices == [0, 1]

    # Make sure collating returns full tensors if no elements are sparse.
    x = torch.ones((1))
    y = 2.0 * torch.ones((1))
    b = [(x, y), (x, y), (x, y), (x, y)]
    x, y = sparse_collate(b)
    assert isinstance(x, torch.Tensor)


def test_training_find_files(cpcir_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    training_data = CIMRDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir"],
    )
    assert training_data.sequence_starts.size == 12



def test_sparse_data(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are set to None.
    """
    training_data = CIMRDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_input_policy="sparse",
        window_size = 128
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)


def test_pretrain_dataset(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that the training data finds the expected start times for
    sampling sequences.
    """
    for i in range(10):
        inputs = ["gmi", "cpcir"]
        training_data = CIMRPretrainDataset(
            cpcir_data,
            reference_data="mrms",
            inputs=inputs,
            missing_input_policy="sparse"
        )
        assert len(training_data.sequence_starts) == 24

        x, y = training_data[0]
        assert x["gmi"] is not None

        x, y = training_data[4]
        assert x["cpcir"] is not None


def test_full_domain(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test loading of full domain data.
    """
    for i in range(10):
        inputs = ["cpcir", "gmi"]
        training_data = CIMRPretrainDataset(
            cpcir_data,
            reference_data="mrms",
            inputs=inputs,
            missing_input_policy="sparse"
        )

        iter = training_data.full_domain()
        time, x, y = next(iter)

        assert x["cpcir"].shape[-2:] == (960, 1920)


def test_missing_input_policies(cpcir_data, gmi_data, mrms_surface_precip_data):
    """
    Test that missing inputs are handled correctly.
    """
    training_data = CIMRDataset(
        cpcir_data,
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        missing_input_policy="sparse",
        window_size = 128
    )
    x, y = training_data[1]
    assert x["gmi"] is None
    assert x["cpcir"].shape[1:] == (128, 128)
