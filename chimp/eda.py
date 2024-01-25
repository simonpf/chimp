"""
chimp.lr_search
===============

Implements learning-rate search for CHIMP retrievals.
"""
import logging
from pathlib import Path
from typing import Optional

import click

from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.eda import run_eda
from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.config import ComputeConfig, InputConfig
from pytorch_retrieve.utils import (
    read_model_config,
    read_training_config,
    read_compute_config,
)

from chimp.training import TrainingConfig


@click.option(
    "--model_path",
    default=None,
    help="The model directory. Defaults to the current working directory",
)
@click.option(
    "--model_config",
    default=None,
    help=(
        "Path to the model config file. If not provided, pytorch_retrieve "
        " will look for a 'model.toml' or 'model.yaml' file in the current "
        " directory."
    ),
)
@click.option(
    "--training_config",
    default=None,
    help=(
        "Path to the training config file. If not provided, pytorch_retrieve "
        " will look for a 'training.toml' or 'training.yaml' file in the current "
        " directory."
    ),
)
def cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
) -> int:
    """
    Train retrieval model.

    This command runs the training of the retrieval model specified by the
    model and training configuration files.

    """
    import chimp.data.seviri
    import chimp.data.gpm
    import chimp.data.goes
    import chimp.data.cpcir
    import chimp.data.baltrad
    import chimp.data.mrms
    import chimp.data.gridsat
    import chimp.data.daily_precip

    if model_path is None:
        model_path = Path(".")

    LOGGER = logging.getLogger(__name__)
    model_config = read_model_config(LOGGER, model_path, model_config)
    if model_config is None:
        return 1
    retrieval_model = compile_architecture(model_config)

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        return 1
    for cfg in training_config.values():
        cfg["sequence_length"] = 1
        cfg["forecast"] = 0

    input_configs = {
        name: InputConfig.parse(name, cfg)
        for name, cfg in model_config["input"].items()
    }
    training_schedule = {
        name: TrainingConfig.parse(name, cfg) for name, cfg in training_config.items()
    }

    run_eda(model_path, input_configs, training_schedule)
