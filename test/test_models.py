"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""

import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import TestDataset
from cimr.models import CIMR, CIMRSeq, Encoder, FPHead, Merger, TimeStepper


def test_cimr_model():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = TestDataset(
        sequence_length=1
    )
    data_loader = DataLoader(
        dataset,
        batch_size = 2
    )

    it = iter(data_loader)
    x, y = next(it)

    cimr = CIMR(n_stages=3, n_features=32, n_outputs=64, n_blocks=2)

    y = cimr(x[0])


def test_cimr_sequence_model():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = TestDataset(
        sequence_length=2
    )
    data_loader = DataLoader(
        dataset,
        batch_size = 2
    )

    it = iter(data_loader)
    x, y = next(it)

    cimr = CIMRSeq(n_stages=3, n_features=32, n_outputs=64, n_blocks=2)
    y = cimr(x)


def test_cimr_encoder():
    encoder = Encoder(8, 1, 16, 8)
    x = torch.normal(0, 1, ((1, 8, 32, 32)))

    y = encoder(x)
    assert len(y) == 5
    assert y[0].shape[-1] == 32
    assert y[-1].shape[-1] == 2


def test_fp_head():
    encoder = Encoder(8, 1, 16, 8)
    x = torch.normal(0, 1, ((1, 8, 32, 32)))
    y = encoder(x)

    head = FPHead(8, 5, 16, 4)
    y = head(y)


def test_cimr_time_stepper():
    stepper = TimeStepper(3, 16)

    x = [
        torch.normal(0, 1, ((1, 16, 32, 32))),
        torch.normal(0, 1, ((1, 16, 16, 16))),
        torch.normal(0, 1, ((1, 16, 8, 8))),
    ]

    y = stepper(x)
    assert len(y) == 3
    assert y[0].shape[-1] == 32
    assert y[-1].shape[-1] == 8
