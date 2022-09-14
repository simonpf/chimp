"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""

import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import TestDataset
from cimr.models import CIMR, CIMRSeq


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
