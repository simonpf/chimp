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
from pytorch_retrieve.lr_search import run_lr_search
from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.config import ComputeConfig
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
@click.option(
    "--compute_config",
    default=None,
    help=(
        "Path to the compute config file defining the compute environment for "
        " the training."
    ),
)
@click.option(
    "--min_lr",
    default=1e-8,
    help=("The smallest learning rate to test."),
)
@click.option(
    "--max_lr",
    default=1e2,
    help=("The largest learning rate to test."),
)
@click.option(
    "--n_steps",
    default=100,
    help=("The number of training steps to perform."),
)
@click.option(
    "--plot",
    default=True,
    help=("Whether or not to show a plot of the results."),
)
def cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    min_lr: float,
    max_lr: float,
    n_steps: int,
    plot: bool,
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
    training_schedule = {
        name: TrainingConfig.parse(name, cfg) for name, cfg in training_config.items()
    }

    module = LightningRetrieval(retrieval_model, training_schedule=training_schedule)

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if isinstance(compute_config, dict):
        compute_config = ComputeConfig.parse(compute_config)

    run_lr_search(
        module,
        compute_config=compute_config,
        min_lr=min_lr,
        max_lr=max_lr,
        n_steps=n_steps,
        plot=plot,
    )
