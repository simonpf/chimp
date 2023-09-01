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
from quantnn.mrnn import MRNN

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
    compile_mrnn,
    load_config,
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


def test_compile_parallel_encoder():
    """
    Test the compilation of an encoder with separate encoders for different
    inputs.
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
        combined=False
    )

    encoder = compile_encoder(
        input_configs=input_configs,
        encoder_config=encoder_config
    )

    x = {
        "gmi": sparse_collate([
            torch.zeros((13, 64, 64), dtype=torch.float32),
            None
        ]),
        "ssmis": sparse_collate([
            None,
            torch.zeros((11, 64, 64), dtype=torch.float32),
        ])
    }

    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[-1].shape == (2, 128, 8, 8)

    x = {
        "gmi": sparse_collate([
            None,
            None
        ]),
        "ssmis": sparse_collate([
            None,
            None
        ])
    }

    assert y is None


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
            "surface_precip",
            "quantile_loss",
        ),
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
    )

    decoder_config = DecoderConfig(
        "convnet",
        channels=[64, 32, 16, 16],
        stage_depths=[1, 1, 1, 1],
        upsampling_factors=[2, 2, 2, 2],
        skip_connections=4
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
            "surface_precip",
            "quantile_loss",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
    )

    decoder_config = DecoderConfig(
        "convnet",
        channels=[64, 32, 16, 16],
        stage_depths=[1, 1, 1, 1],
        upsampling_factors=[2, 2, 2, 2],
        skip_connections=4
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
    assert "surface_precip" in y
    assert y["surface_precip"].shape == (1, 32, 128, 128)


def test_compile_mrnn():
    """
    Test the compilation of the quantnn MRNN.
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
            "surface_precip",
            "quantile_loss",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
    )

    decoder_config = DecoderConfig(
        "convnet",
        channels=[64, 32, 16, 16],
        stage_depths=[1, 1, 1, 1],
        upsampling_factors=[2, 2, 2, 2],
        skip_connections=4
    )
    model_config = ModelConfig(
        input_configs,
        output_configs,
        encoder_config,
        decoder_config
    )

    mrnn = compile_mrnn(model_config)
    assert isinstance(mrnn, MRNN)


def test_load_config():
    """
    Test the loading of a pre-defined configuration.
    """
    config = load_config("gremlin")
    assert config.encoder_config.channels == [32, 32, 32, 32]

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    model = compile_model(config)
    assert model.n_params == 46593


def test_gremlin():
    """
    Test the loading of a pre-defined configuration.
    """
    config = load_config("gremlin")
    assert config.encoder_config.channels == [32, 32, 32, 32]

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    model = compile_model(config)

    x = {
        "cpcir": torch.ones(
            (1, 1, 128, 128)
        )
    }
    y = model(x)
    assert y["surface_precip"].shape == (1, 128, 128)


def test_resnet18():
    """
    Test the loading of the ResNet18 configuration.
    """
    config = load_config("resnet18")

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse"
        ),
    ]

    model = compile_model(config)

    x = {
        "cpcir": torch.ones(
            (1, 1, 256, 256)
        )
    }
    y = model(x)
    assert y["surface_precip"].shape == (1, 256, 256)


def test_convnext18():
    """
    Test the loading of the ConvNext18 configuration.
    """
    config = load_config("resnext18")

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse"
        ),
    ]

    model = compile_model(config)

    x = {
        "cpcir": torch.ones(
            (1, 1, 256, 256)
        )
    }
    y = model(x)
    assert y["surface_precip"].shape == (1, 256, 256)


def test_resnext18():
    """
    Test the loading of the ConvNext18 configuration.
    """
    config = load_config("resnext18")

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse"
        ),
    ]

    model = compile_model(config)

    x = {
        "cpcir": torch.ones(
            (1, 1, 256, 256)
        )
    }
    y = model(x)
    assert y["surface_precip"].shape == (1, 256, 256)


def test_dlax18():
    """
    Test the loading of the DLAX18 configuration.
    """
    config = load_config("dlax18")

    config.input_configs = [
        InputConfig(
            inputs.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    config.output_configs = [
        OutputConfig(
            reference.MRMS,
            "surface_precip",
            "mse"
        ),
    ]

    model = compile_model(config)

    x = {
        "cpcir": torch.ones(
            (1, 1, 256, 256)
        )
    }
    y = model(x)
    assert y["surface_precip"].shape == (1, 256, 256)
