"""
Tests for neural network models defined in the ``cimr.models``
sub-module.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cimr.data.training_data import (CIMRDataset,
                                     SuperpositionDataset,
                                     sparse_collate)
from cimr.config import (
    InputConfig,
    OutputConfig,
    EncoderConfig,
    DecoderConfig,
    ModelConfig
)
from cimr.data import inputs, reference
from cimr.models import (
    compile_encoder,
    compile_decoder,
    compile_model,
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


def test_compile_encoder():
    """
    Test the compilation of the encoder.
    """
    input_configs = [
        InputConfig(
            inputs.GMI,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=2
        ),
        InputConfig(
            inputs.SSMIS,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=1
        )
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
        skip_connections=True
    )

    encoder = compile_encoder(
        input_configs=input_configs,
        encoder_config=encoder_config
    )

    x = {
        "gmi": torch.zeros(
            (1, 13, 64, 64),
            dtype=torch.float32
        ),
        "ssmis": torch.zeros(
            (1, 11, 64, 64),
            dtype=torch.float32
        )
    }

    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[-1].shape == (1, 128, 8, 8)


def test_compile_decoder():
    """
    Test the compilation of the decoder.
    """
    input_configs = [
        InputConfig(
            inputs.GMI,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=2
        ),
        InputConfig(
            inputs.SSMIS,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=1
        )
    ]
    output_configs = [
        OutputConfig(
            reference.MRMS,
            "quantile_loss",
        ),
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
        skip_connections=True
    )

    decoder_config = DecoderConfig(
        "convnet",
        channels=[64, 32, 16, 16, 16],
        stage_depths=[1, 1, 1, 1, 1],
        upsampling_factors=[2, 2, 2, 2, 2],
    )

    encoder = compile_encoder(
        input_configs=input_configs,
        encoder_config=encoder_config
    )
    decoder = compile_decoder(
        input_configs=input_configs,
        output_configs=output_configs,
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )

    x = {
        "gmi": torch.zeros(
            (1, 13, 64, 64),
            dtype=torch.float32
        ),
        "ssmis": torch.zeros(
            (1, 11, 64, 64),
            dtype=torch.float32
        )
    }

    y = encoder(x, return_skips=True)
    print("SC :", decoder.skip_connections)
    for tensor in y:
        print(tensor.shape)
    y = decoder(y)
    assert y.shape == (1, 16, 128, 128)


def test_compile_model():
    """
    Test the compilation of the full CIMRModel.
    """
    input_configs = [
        InputConfig(
            inputs.GMI,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=2
        ),
        InputConfig(
            inputs.SSMIS,
            stem_depth=2,
            stem_kernel_size=7,
            stem_downsampling=1
        )
    ]
    output_configs = [
        OutputConfig(
            reference.MRMS,
            "quantile_loss",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
        skip_connections=True
    )

    decoder_config = DecoderConfig(
        "convnet",
        channels=[64, 32, 16, 16],
        stage_depths=[1, 1, 1, 1],
        upsampling_factors=[2, 2, 2, 2],
    )
    model_config = ModelConfig(
        input_configs,
        output_configs,
        encoder_config,
        decoder_config
    )

    cimr = compile_model(model_config)


    x = {
        "gmi": torch.zeros(
            (1, 13, 64, 64),
            dtype=torch.float32
        ),
        "ssmis": torch.zeros(
            (1, 11, 64, 64),
            dtype=torch.float32
        )
    }

    y = cimr(x)

    assert len(y) == 1
    assert "mrms" in y
    assert y["mrms"].shape == (1, 32, 128, 128)



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
