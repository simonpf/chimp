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


@dataclass
class TrainingConfig(pr.training.TrainingConfigBase):
    """
    A dataclass to hold parameters of a single training stage.
    """

    training_data_path: Path
    validation_data_path: Path
    input_datasets: List[str]
    reference_datasets: List[str]
    sample_rate: int
    sequence_length: int
    forecast: int
    forecast_range: Optional[int]
    shrink_output: Optional[int]
    scene_size: int
    augment: bool
    time_step: int
    n_epochs: int
    batch_size: int
    optimizer: str
    optimizer_args: Optional[dict] = None
    scheduler: str = None
    scheduler_args: Optional[dict] = None
    milestones: Optional[list] = None
    gradient_clipping: Optional[float] = None
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False
    metrics: Optional[Dict[str, List["Metric"]]] = None
    include_input_steps: bool = False
    require_input: bool = False

    log_every_n_steps: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    accumulate_grad_batches: Optional[int] = None
    n_data_loader_workers: int = 12
    load_weights: Optional[str] = None

    @classmethod
    def parse(cls, name, config_dict: Dict[str, object]):
        """
        Parses a single training stage from a dictionary of training settings.

        Args:
            name: Name of the training stage.
            config_dict: The dictionary containing the training settings.

        Return:
            A TrainingConfig object containing the settings from the dictionary.
        """
        training_data_path = get_config_attr(
            "training_data_path", str, config_dict, f"training stage '{name}'"
        )
        validation_data_path = get_config_attr(
            "validation_data_path",
            str,
            config_dict,
            f"training stage '{name}'",
            required=False,
        )
        input_datasets = get_config_attr(
            "input_datasets", list, config_dict, f"training stage '{name}'"
        )
        reference_datasets = get_config_attr(
            "reference_datasets", list, config_dict, f"training stage '{name}'"
        )
        sample_rate = get_config_attr(
            "sample_rate", float, config_dict, f"training stage '{name}'", 1.0
        )
        sequence_length = get_config_attr(
            "sequence_length", int, config_dict, f"training stage '{name}'", 1
        )
        forecast = get_config_attr(
            "forecast", int, config_dict, f"training stage '{name}'", 0
        )
        forecast_range = get_config_attr(
            "forecast_range", int, config_dict, f"training stage '{name}'", None
        )
        shrink_output = get_config_attr(
            "shrink_output", int, config_dict, f"training stage '{name}'", None
        )
        scene_size = get_config_attr(
            "scene_size", int, config_dict, f"training stage '{name}'", 128
        )
        augment = get_config_attr(
            "augment", bool, config_dict, f"training stage '{name}'", True
        )
        time_step = get_config_attr(
            "time_step", str, config_dict, f"training stage '{name}'", required=False
        )

        dataset_module = get_config_attr(
            "dataset_module", str, config_dict, f"training stage '{name}'"
        )
        training_dataset = get_config_attr(
            "training_dataset", str, config_dict, f"training stage '{name}'"
        )
        training_dataset_args = get_config_attr(
            "training_dataset_args", dict, config_dict, f"training stage '{name}'"
        )
        validation_dataset = get_config_attr(
            "validation_dataset",
            str,
            config_dict,
            f"training stage '{name}'",
            training_dataset,
        )
        validation_dataset_args = get_config_attr(
            "validation_dataset_args", dict, config_dict, f"training stage '{name}'", ""
        )
        if validation_dataset_args == "":
            validation_dataset_args = None

        n_epochs = get_config_attr(
            "n_epochs", int, config_dict, f"training stage '{name}'"
        )
        batch_size = get_config_attr(
            "batch_size", int, config_dict, f"training stage '{name}'"
        )

        optimizer = get_config_attr(
            "optimizer", str, config_dict, f"training stage '{name}'", required=True
        )
        optimizer_args = get_config_attr(
            "optimizer_args", dict, config_dict, f"training stage '{name}'", {}
        )

        scheduler = get_config_attr(
            "scheduler", None, config_dict, f"training stage {name}", None
        )
        scheduler_args = get_config_attr(
            "scheduler_args", None, config_dict, f"training stage {name}", {}
        )
        milestones = get_config_attr(
            "milestones", list, config_dict, f"training stage {name}", None
        )
        gradient_clipping = get_config_attr(
            "gradient_clipping", float, config_dict, f"training stage '{name}'", -1.0
        )
        if gradient_clipping < 0:
            gradient_clipping = None

        minimum_lr = get_config_attr(
            "minimum_lr", float, config_dict, f"training stage '{name}'", -1.0
        )
        if minimum_lr < 0:
            minimum_lr = None

        reuse_optimizer = get_config_attr(
            "reuse_optimizer", bool, config_dict, f"training stage '{name}'", False
        )
        stepwise_scheduling = get_config_attr(
            "stepwise_scheduling", bool, config_dict, f"training stage '{name}'", False
        )

        metrics = config_dict.get("metrics", [])

        include_input_steps = get_config_attr(
            "include_input_steps", bool, config_dict, f"training stage '{name}'", forecast == 0
        )
        require_input = get_config_attr(
            "require_input", bool, config_dict, f"training stage '{name}'", False
        )

        log_every_n_steps = config_dict.get("log_every_n_steps", -1)
        if log_every_n_steps < 0:
            if n_epochs < 100:
                log_every_n_steps = 1
            else:
                log_every_n_steps = 50

        gradient_clip_val = get_config_attr(
            "gradient_clip_val", float, config_dict, f"training stage {name}", None
        )
        gradient_clip_algorithm = get_config_attr(
            "gradient_clip_algorithm", str, config_dict, f"training stage {name}", None
        )
        accumulate_grad_batches = get_config_attr(
            "accumulate_grad_batches", int, config_dict, f"training stage {name}", 1
        )
        n_data_loader_workers = get_config_attr(
            "n_data_loader_workers", int, config_dict, f"training stage {name}", 12
        )
        load_weights = get_config_attr(
            "load_weights", str, config_dict, f"training stage {name}", None
        )

        return TrainingConfig(
            training_data_path=training_data_path,
            validation_data_path=validation_data_path,
            input_datasets=input_datasets,
            reference_datasets=reference_datasets,
            sample_rate=sample_rate,
            sequence_length=sequence_length,
            forecast=forecast,
            forecast_range=forecast_range,
            shrink_output=shrink_output,
            scene_size=scene_size,
            augment=augment,
            time_step=time_step,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            milestones=milestones,
            batch_size=batch_size,
            gradient_clipping=gradient_clipping,
            minimum_lr=minimum_lr,
            reuse_optimizer=reuse_optimizer,
            stepwise_scheduling=stepwise_scheduling,
            metrics=metrics,
            include_input_steps=include_input_steps,
            require_input=require_input,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            accumulate_grad_batches=accumulate_grad_batches,
            n_data_loader_workers=n_data_loader_workers,
            load_weights=load_weights,
        )

    def get_training_dataset(
        self,
    ) -> object:
        """
        Instantiates the appropriate training dataset.
        """
        if self.sequence_length == 1:
            return SingleStepDataset(
                self.training_data_path,
                input_datasets=self.input_datasets,
                reference_datasets=self.reference_datasets,
                sample_rate=self.sample_rate,
                augment=self.augment,
                scene_size=self.scene_size,
            )
        else:
            return SequenceDataset(
                self.training_data_path,
                input_datasets=self.input_datasets,
                reference_datasets=self.reference_datasets,
                sample_rate=self.sample_rate,
                scene_size=self.scene_size,
                sequence_length=self.sequence_length,
                forecast=self.forecast,
                forecast_range=self.forecast_range,
                shrink_output=self.shrink_output,
                augment=self.augment,
                include_input_steps=self.include_input_steps,
                require_input=self.require_input
            )

    def get_validation_dataset(self):
        """
        Instantiates the appropriate validation dataset.
        """
        if self.validation_data_path is None:
            return None
        if self.sequence_length == 1:
            return SingleStepDataset(
                self.validation_data_path,
                input_datasets=self.input_datasets,
                reference_datasets=self.reference_datasets,
                sample_rate=self.sample_rate,
                augment=False,
                scene_size=self.scene_size,
                validation=True,
            )
        else:
            return SequenceDataset(
                self.validation_data_path,
                input_datasets=self.input_datasets,
                reference_datasets=self.reference_datasets,
                sample_rate=self.sample_rate,
                scene_size=self.scene_size,
                sequence_length=self.sequence_length,
                forecast=self.forecast,
                forecast_range=self.forecast_range,
                shrink_output=self.shrink_output,
                augment=False,
                validation=True,
                include_input_steps=self.include_input_steps,
                require_input=self.require_input
            )


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
