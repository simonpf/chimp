import pytest

from cimr.configs import parse_model_config

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

    assert len(model_config.output_configs) == 1
    output_config = model_config.output_configs[0]
    assert output_config.loss == "quantile_loss"
    assert output_config.shape == (28,)

    encoder_config = model_config.encoder_config
    assert encoder_config.block_type == "resnet"
    assert encoder_config.stage_depths == [2, 3, 3, 2]
    assert encoder_config.downsampling_factors == [2, 2, 2, 2]
    assert encoder_config.skip_connections == False

    decoder_config = model_config.decoder_config
    assert decoder_config.block_type == "resnet"
    assert decoder_config.stage_depths == [1, 1, 1, 1]
