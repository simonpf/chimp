"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""

import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import (SuperpositionDataset,
                                     sparse_collate)
from cimr.models import (
    CIMRBaseline,
    CIMRSeq,
    TimeStepper,
    CIMRBaselineV2,
    CIMRBaselineV3
)



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
    cimr = CIMRBaseline(n_stages=3, stage_depths=2)
    y = cimr(x)

    # Test with subset of inputs.
    for source in ["geo", "visir", "mw"]:
        cimr = CIMRBaseline(n_stages=4, sources=[source], stage_depths=2)
        y = cimr(x)
        assert y.shape[-2] == 128
        assert y.shape[-1] == 128


def test_cimr_model_2():
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
    cimr = CIMRBaselineV2(n_stages=3, stage_depths=2)
    y = cimr(x)

    # Test with subset of inputs.
    for source in ["geo", "visir", "mw"]:
        cimr = CIMRBaseline(n_stages=4, sources=[source], stage_depths=2)
        y = cimr(x)
        assert y.shape[-2] == 128
        assert y.shape[-1] == 128


def test_cimr_model_3():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = SuperpositionDataset(
        size=128,
        n_steps=1,
        sparse=True
    )
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=sparse_collate
    )
    it = iter(data_loader)
    x, y = next(it)

    # Test with all inputs.
    cimr = CIMRBaselineV3()

    y = cimr(x)

    ## Test with subset of inputs.
    #for source in ["geo", "visir", "mw"]:
    #    cimr = CIMRBaselineV3(sources=[source])
    #    y = cimr(x)
    #    assert y.shape[-2] == 128
    #    assert y.shape[-1] == 128


def test_cimr_sequence_model():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    dataset = SuperpositionDataset(
        size=256,
        n_steps=2
    )
    data_loader = DataLoader(
        dataset,
        batch_size = 2
    )
    it = iter(data_loader)
    x, y = next(it)

    cimr = CIMRSeq(n_stages=5, stage_depths=1, aggregator_type="block")
    y = cimr(x)
