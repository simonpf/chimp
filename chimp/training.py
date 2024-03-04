"""
chimp.training
=============

Module implementing training functionality.
"""
import click
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.config import get_config_attr, ComputeConfig
from pytorch_retrieve.utils import (
    read_model_config,
    read_training_config,
    read_compute_config,
)
import pytorch_retrieve as pr
from pytorch_retrieve import metrics
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.training import run_training
from torch.utils.data import DataLoader

from chimp.data.training_data import SingleStepDataset, SequenceDataset


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
    "--resume",
    "-r",
    "resume",
    is_flag=True,
    default=False,
    help=("If set, training will continue from a checkpoint file if available."),
)
def cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    resume: bool = False,
) -> int:
    """
    Train a retrieval model.

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

    module = LightningRetrieval(
        retrieval_model,
        name="retrieval_module",
        training_schedule=training_schedule
    )

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if compute_config is not None:
        compute_config = ComputeConfig.parse(compute_config)

    checkpoint = None
    if resume:
        checkpoint = find_most_recent_checkpoint(
            model_path / "checkpoints", module.name
        )

    run_training(
        model_path,
        module,
        compute_config=compute_config,
        checkpoint=checkpoint,
    )
