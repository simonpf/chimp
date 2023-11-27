import pytest

import torch
from torch.utils.data import DataLoader

from chimp.models.encoders import compile_encoder
from chimp.config import InputConfig, EncoderConfig
from chimp.data import get_input
from chimp.data.training_data import SingleStepDataset


@pytest.fixture
def single_scale_shared_encoder():
    input_configs = [
        InputConfig(
            get_input("cpcir"), stem_depth=1, stem_kernel_size=3, stem_downsampling=1
        ),
        InputConfig(
            get_input("mhs"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("ssmis"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("gmi"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("atms"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("amsr2"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
    ]
    encoder_config = EncoderConfig(
        "resnext",
        channels=[64, 128, 256, 512, 512, 512],
        stage_depths=[2, 2, 4, 4, 2, 2],
        downsampling_factors=[2, 2, 2, 2, 2],
        combined=True,
        multi_scale=False,
    )
    return input_configs, encoder_config


def test_single_scale_shared_encoder(single_scale_shared_encoder, training_data_multi):
    input_cfgs, encoder_cfg = single_scale_shared_encoder
    encoder = compile_encoder(input_cfgs, encoder_cfg)

    data_loader = DataLoader(training_data_multi, batch_size=1)
    x, y = next(iter(data_loader))

    encs = encoder(x, return_skips=True)
    assert isinstance(encs, dict)


@pytest.fixture
def multi_scale_shared_encoder():
    input_configs = [
        InputConfig(
            get_input("cpcir"), stem_depth=1, stem_kernel_size=3, stem_downsampling=1
        ),
        InputConfig(
            get_input("mhs"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("ssmis"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("gmi"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("atms"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("amsr2"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
    ]
    encoder_config = EncoderConfig(
        "resnext",
        channels=[64, 128, 256, 512, 512, 512],
        stage_depths=[2, 2, 4, 4, 2, 2],
        downsampling_factors=[2, 2, 2, 2, 2],
        combined=True,
        multi_scale=True,
    )
    return input_configs, encoder_config


def test_multi_scale_shared_encoder(multi_scale_shared_encoder, training_data_multi):
    input_cfgs, encoder_cfg = multi_scale_shared_encoder
    encoder = compile_encoder(input_cfgs, encoder_cfg)

    data_loader = DataLoader(training_data_multi, batch_size=1)
    x, y = next(iter(data_loader))

    encs = encoder(x, return_skips=True)

    assert isinstance(encs, dict)
    for chans, enc in zip(encoder_cfg.channels, encs.values()):
        assert chans == enc.shape[1]

    assert len(encoder.encoder.stage_inputs[0]) == 3
    assert len(encoder.encoder.stages) == len(encoder_cfg.stage_depths)


@pytest.fixture
def multi_scale_parallel_encoder():
    input_configs = [
        InputConfig(
            get_input("cpcir"), stem_depth=1, stem_kernel_size=3, stem_downsampling=1
        ),
        InputConfig(
            get_input("mhs"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("ssmis"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("gmi"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("atms"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("amsr2"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
    ]
    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
        combined=False,
        multi_scale=True,
    )
    return input_configs, encoder_config


def test_multi_scale_parallel_encoder(
    multi_scale_parallel_encoder, training_data_multi
):
    input_cfgs, encoder_cfg = multi_scale_parallel_encoder
    encoder = compile_encoder(input_cfgs, encoder_cfg)

    data_loader = DataLoader(training_data_multi, batch_size=1)
    x, y = next(iter(data_loader))

    encs = encoder(x, return_skips=True)
    for chans, enc in zip(encoder_cfg.channels, encs.values()):
        assert chans == enc.shape[1]
    assert isinstance(encs, dict)
    assert len(encoder.encoders) == 6
    assert len(encoder.encoders["cpcir"].stages) == len(encoder_cfg.stage_depths)


@pytest.fixture
def single_scale_parallel_encoder():
    input_configs = [
        InputConfig(
            get_input("cpcir"), stem_depth=1, stem_kernel_size=3, stem_downsampling=1
        ),
        InputConfig(
            get_input("mhs"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("ssmis"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("gmi"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("atms"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
        InputConfig(
            get_input("amsr2"), stem_depth=2, stem_kernel_size=7, stem_downsampling=1
        ),
    ]
    encoder_config = EncoderConfig(
        "convnet",
        channels=[16, 32, 64, 128],
        stage_depths=[2, 2, 4, 4],
        downsampling_factors=[2, 2, 2],
        combined=False,
        multi_scale=False,
    )
    return input_configs, encoder_config


def test_single_scale_parallel_encoder(
    single_scale_parallel_encoder, training_data_multi
):
    input_cfgs, encoder_cfg = single_scale_parallel_encoder
    encoder = compile_encoder(input_cfgs, encoder_cfg)

    data_loader = DataLoader(training_data_multi, batch_size=1)
    x, y = next(iter(data_loader))

    encs = encoder(x, return_skips=True)
    for chans, enc in zip(encoder_cfg.channels, encs.values()):
        assert chans == enc.shape[1]
    assert isinstance(encs, dict)
    assert len(encoder.encoders) == 6
    for enc in encoder.encoders.values():
        assert len(enc.stages) == len(encoder_cfg.stage_depths)
