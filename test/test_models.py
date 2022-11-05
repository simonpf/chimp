"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""

import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import SuperpositionDataset
from cimr.models import CIMRNaive, CIMRSeqNaive, TimeStepper


def test_cimr_model():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = SuperpositionDataset(
        size=128,
        n_steps=1
    )
    data_loader = DataLoader(
        dataset,
        batch_size=2
    )
    it = iter(data_loader)
    x, y = next(it)

    # Test with all inputs.
    cimr = CIMRNaive(n_stages=3, stage_depths=2)
    y = cimr(x)

    # Test with subset of inputs.
    for source in ["geo", "visir", "mw"]:
        cimr = CIMRNaive(n_stages=4, sources=[source], stage_depths=2)
        y = cimr(x)
        assert y.shape[-2] == 128
        assert y.shape[-1] == 128


def test_cimr_sequence_model():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = SuperpositionDataset(
        size=128,
        n_steps=2
    )
    data_loader = DataLoader(
        dataset,
        batch_size = 2
    )
    it = iter(data_loader)
    x, y = next(it)

    cimr = CIMRSeqNaive(n_stages=5, stage_depths=1, aggregator_type="block")
    y = cimr(x)
