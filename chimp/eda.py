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
from pytorch_retrieve.config import ComputeConfig, InputConfig, OutputConfig
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
    "--stats_path",
    default=None,
    help=(
        "Directory to which to write the resulting statistics files. If not "
        "set, they will be written to directory named 'stats' in the model "
        "path. "
    )
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
@click.option(
    "--compute_config",
    default=None,
    help=(
        "Path to the compute config file defining the compute environment for "
        " the training."
    ),
)
@click.option(
    "--stage",
    default=None,
    help=(
        "If provided, training settings for the EDA will be loaded from this "
        "stage of the training schedule."
    )
)
def cli(
    model_path: Optional[Path],
    stats_path: Path,
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    stage: Optional[str]
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
    import chimp.data.opera
    import chimp.data.mrms
    import chimp.data.gridsat
    import chimp.data.daily_precip

    if model_path is None:
        model_path = Path(".")
    else:
        model_path = Path(model_path)

    if stats_path is None:
        stats_path = model_path / "stats"

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
    output_configs = {
        name: OutputConfig.parse(name, cfg)
        for name, cfg in model_config["output"].items()
    }
    training_schedule = {
        name: TrainingConfig.parse(name, cfg) for name, cfg in training_config.items()
    }
    if stage is None:
        training_config = next(iter(training_schedule.values()))
    else:
        if stage not in training_schedule:
            LOGGER.error(
                "The given stage '%s' is not a stage in the provided training "
                "schedule."
            )
        training_config = training_schedule[stage]

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if isinstance(compute_config, dict):
        compute_config = ComputeConfig.parse(compute_config)

    run_eda(
        stats_path,
        input_configs,
        output_configs,
        training_schedule,
        compute_config
    )
