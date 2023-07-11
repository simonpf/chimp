"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import (CIMRDataset,
                                     SuperpositionDataset,
                                     sparse_collate)
from cimr.models import (
    parse_model_config,
    CIMRBaseline,
    CIMRSeq,
    TimeStepper,
    CIMRBaselineV2,
    CIMRBaselineV3
)


TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)


MODEL_CONFIG = """
[gmi]
type = input
name = gmi
stem_type = patchify
stem_depth = 2
stem_downsampling = 3

[mhs]
type = input
name = mhs
stem_type = standard
stem_depth = 1

[mrms]
type = output
name = mrms
loss = quantile_loss
quantiles = np.linspace(0, 1, 34)[1:-1]
shape = (28,)


[encoder]
type = encoder
block_type = resnet
stage_depths = 2 3 3 2
downsampling_factors = 2 2 2 2
skip_connections = False

[decoder]
type = decoder
block_type = resnet
stage_depths = 1 1 1 1
"""

def test_parse_model_config(tmp_path):
    """
    Test parsing of the model config.
    """
    with open(tmp_path / "cimr_model.ini", "w") as config_file:
        config_file.write(MODEL_CONFIG)

    model_config = parse_model_config(tmp_path / "cimr_model.ini")

    assert len(model_config.input_configs) == 2
    input_config = model_config.input_configs[0]
    assert input_config.stem_depth == 2
    assert input_config.stem_type == "patchify"
    assert input_config.stem_downsampling == 3

    encoder_config = model_config.encoder_config
    assert encoder_config.block_type == "resnet"
    assert encoder_config.stage_depths == [2, 3, 3, 2]
    assert encoder_config.downsampling_factors == [2, 2, 2, 2]
    assert encoder_config.skip_connections == False

    decoder_config = model_config.decoder_config
    assert decoder_config.block_type == "resnet"
    assert decoder_config.stage_depths == [1, 1, 1, 1]



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
    inputs = ["cpcir", "gmi"]
    reference_data = "mrms"
    training_data = CIMRDataset(
        TEST_DATA / "training_data",
        reference_data=reference_data,
        inputs=inputs,
        sparse=True
    )
    data_loader = DataLoader(
        training_data,
        batch_size=8,
        collate_fn=sparse_collate
    )
    it = iter(data_loader)
    x, y = next(it)

    # Test with all inputs.
    cimr = CIMRBaselineV2(inputs=inputs, n_stages=3, stage_depths=2)
    y = cimr(x)

    # Test with subset of inputs.
    for inpt in inputs:
        cimr = CIMRBaselineV2(n_stages=4, inputs=[inpt], stage_depths=2)
        y = cimr(x)["surface_precip"]
        assert y.shape[-2] == 128
        assert y.shape[-1] == 128


def test_cimr_model_3():
    """
    Test propagation of inputs through CIMR model to ensure that the
    architecture is sound.
    """
    inputs = ["cpcir", "gmi", "atms", "mhs"]
    reference_data = "mrms"
    training_data = CIMRDataset(
        TEST_DATA / "training_data",
        reference_data=reference_data,
        inputs=inputs,
        sparse=True
    )
    data_loader = DataLoader(
        training_data,
        batch_size=8,
        collate_fn=sparse_collate
    )
    it = iter(data_loader)
    x, y = next(it)

    cimr = CIMRBaselineV3(inputs=inputs, reference_data=reference_data)

    print(cimr.encoder.stems)

    y = cimr(x)



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
