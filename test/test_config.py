"""
Tests for the cimr.config module
"""

import pytest
import torch
from torch import nn

from cimr.config import (
    parse_model_config,
    parse_training_config
)
from cimr.training import get_optimizer_and_scheduler


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
reference_data = mrms
variable = surface_precip
loss = quantile_loss
quantiles = np.linspace(0, 1, 34)[1:-1]
shape = (28,)

[encoder]
type = encoder
block_type = resnet
stage_depths = 2 3 3 2
channels = 16 32 64 128
downsampling_factors = 2 2 2
skip_connections = False

[decoder]
type = decoder
block_type = resnet
channels = 64 32 16 16
stage_depths = 1 1 1 1
upsampling_factors = 2 2 2 2

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

    assert len(model_config.output_configs) == 1
    output_config = model_config.output_configs[0]
    assert output_config.loss == "quantile_loss"
    assert output_config.shape == (28,)

    encoder_config = model_config.encoder_configs[0]
    assert encoder_config.block_type == "resnet"
    assert encoder_config.stage_depths == [2, 3, 3, 2]
    assert encoder_config.downsampling_factors == [2, 2, 2]
    assert encoder_config.skip_connections == False

    decoder_config = model_config.decoder_config
    assert decoder_config.block_type == "resnet"
    assert decoder_config.stage_depths == [1, 1, 1, 1]


TRAINING_CONFIG = """
[first_run]
n_epochs = 20
optimizer = SGD
optimizer_kwargs = {"lr": 1e-3}
"""

def test_parse_training_config(tmp_path):
    """
    Test parsing of a training config.
    """
    with open(tmp_path / "training.ini", "w") as config_file:
        config_file.write(TRAINING_CONFIG)

    training_config = parse_training_config(tmp_path / "training.ini")

    assert len(training_config) == 1
    assert training_config[0].n_epochs == 20

    model = nn.Conv2d(10, 10, 3)

    opt, scheduler, clbks = get_optimizer_and_scheduler(
        training_config[0],
        model
    )

    assert isinstance(opt, torch.optim.SGD)
    assert scheduler is None
    assert len(clbks) == 0
