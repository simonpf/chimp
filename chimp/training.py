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


class ResetParameters(Callback):
    """
    Pytorch lightning callback to reset model parameters.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Args:
             training: The Pytorch lightning training.
             pl_module: The Pytroch lightning module.
        """
        mrnn = pl_module.qrnn

        def reset_params(layer):
            """
            Rest parameters in network layer.
            """
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        mrnn.model.apply(reset_params)


def get_optimizer_and_scheduler(training_config, model, previous_optimizer=None):
    """
    Return torch optimizer, learning-rate scheduler and callback objects
    corresponding to this configuration.

    Args:
        training_config: A TrainingConfig object specifying training
            settings for one training stage.
        model: The model to be trained as a torch.nn.Module object.
        previous_optimizer: Optimizer from the previous stage in case
            it is reused.

    Return:
        A tuple ``(optimizer, scheduler, callbacks)`` containing a PyTorch
        optimizer object ``optimizer``, the corresponding LR scheduler
        ``scheduler`` and a list of callbacks.

    Raises:
        Value error if training configuration specifies to reuse the optimizer
        but 'previous_optimizer' is none.

    """
    if training_config.reuse_optimizer:
        if previous_optimizer is None:
            raise RuntimeError(
                "Training stage '{training_config.name}' has 'reuse_optimizer' "
                "set to 'True' but no previous optimizer is available."
            )
        optimizer = previous_optimizer

    else:
        optimizer_cls = getattr(torch.optim, training_config.optimizer)
        optimizer = optimizer_cls(
            model.parameters(), **training_config.optimizer_args
        )

    scheduler = training_config.scheduler
    if scheduler is None:
        return optimizer, None, []

    if scheduler == "lr_search":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=2.0)
        callbacks = [
            ResetParameters(),
        ]
        return optimizer, scheduler, callbacks

    scheduler = getattr(torch.optim.lr_scheduler, training_config.scheduler)
    scheduler_args = training_config.scheduler_args
    if scheduler_args is None:
        scheduler_args = {}
    scheduler = scheduler(
        optimizer=optimizer,
        **scheduler_args,
    )
    scheduler.stepwise = training_config.stepwise_scheduling

    if training_config.minimum_lr is not None:
        callbacks = [
            EarlyStopping(
                f"Learning rate",
                stopping_threshold=training_config.minimum_lr * 1.001,
                patience=training_config.n_epochs,
                verbose=True,
                strict=True,
            )
        ]
    else:
        callbacks = []

    return optimizer, scheduler, callbacks


def create_data_loaders(
    model_config: "chimp.config.ModelConfig",
    training_config: "chimp.config.TrainingConfig",
    training_data_path: Path,
    validation_data_path: Optional[Path],
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create pytorch Dataloaders for training and validation data.

    Args:
        model_config: Dataclass specifying the model configuration. This
            is required to infer which input and reference data to load.
        training_config: Dataclass specifying the training configuration,
            which defines how many processes to use for the data loading.
        training_data_path: The path pointing to the folder containing
            the training data.
        validation_data_path: The path pointing to the folder containing
            the validation data.
    """
    inputs = []
    for inpt_cfg in model_config.input_configs:
        inputs.append(inpt_cfg.input_data)

    reference_data = None
    for output_cfg in model_config.output_configs:
        if reference_data is None:
            reference_data = output_cfg.reference_data
        else:
            if reference_data != output_cfg.reference_data:
                raise ValueError(
                    "Retrieval outputs must all come from the same " " reference data."
                )

    if model_config.temporal_merging:
        training_data = SequenceDataset(
            training_data_path,
            inputs=inputs,
            reference_data=reference_data,
            sample_rate=training_config.sample_rate,
            sequence_length=training_config.sequence_length,
            forecast=training_config.forecast,
            scene_size=training_config.input_size,
            quality_threshold=training_config.quality_threshold,
            missing_value_policy=training_config.missing_value_policy,
        )
    else:
        training_data = SingleStepDataset(
            training_data_path,
            inputs=inputs,
            reference_data=reference_data,
            sample_rate=training_config.sample_rate,
            sequence_length=training_config.sequence_length,
            scene_size=training_config.input_size,
            quality_threshold=training_config.quality_threshold,
            missing_value_policy=training_config.missing_value_policy,
        )

    training_loader = DataLoader(
        training_data,
        shuffle=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=training_data.init_rng,
        pin_memory=True,
        # collate_fn=sparse_collate
    )
    if validation_data_path is None:
        return training_loader, None

    if model_config.temporal_merging:
        validation_data = SequenceDataset(
            validation_data_path,
            inputs=inputs,
            reference_data=reference_data,
            sample_rate=training_config.sample_rate,
            sequence_length=training_config.sequence_length,
            forecast=training_config.forecast,
            scene_size=training_config.input_size,
            quality_threshold=training_config.quality_threshold,
            missing_value_policy=training_config.missing_value_policy,
            validation=True,
        )
    else:
        validation_data = SingleStepDataset(
            validation_data_path,
            inputs=inputs,
            reference_data=reference_data,
            sample_rate=training_config.sample_rate,
            sequence_length=training_config.sequence_length,
            scene_size=training_config.input_size,
            quality_threshold=training_config.quality_threshold,
            missing_value_policy=training_config.missing_value_policy,
            validation=True,
        )

    validation_loader = DataLoader(
        validation_data,
        shuffle=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=validation_data.init_rng,
        pin_memory=True,
    )
    return training_loader, validation_loader


def train(
    model_name,
    mrnn,
    training_configs,
    training_data_path,
    validation_data_path,
    output_path,
    ckpt_path=None,
):
    """
    Train a CHIMP retrieval model using Pytorch lightning.

    Args:
        model_name:
        mrnn: The 'quantnn.mrnn.MRNN' model implementing the retrieval
            model to use.
        training_configs: A list of TrainingConfig object describing the
            training regime for each training pass.
        training_loader: A pytorch dataloader providing access to the
            training data.
        validation_laoder: A pytorch dataloader providing access to the
            validation data.
        output_path: A path pointing to the directory to which to write
            the trained model.
        ckpt_path: An optional path to a checkpoint from which to resume
            training.
    """
    output_path.mkdir(exist_ok=True, parents=True)

    mtrcs = [
        metrics.Bias(),
        metrics.Correlation(),
        metrics.CRPS(),
        metrics.MeanSquaredError(),
    ]

    lightning_module = mrnn.lightning(
        mask=-100, metrics=mtrcs, name=model_name, log_dir=output_path / "logs"
    )
    if ckpt_path is not None:
        ckpt_data = torch.load(ckpt_path)
        stage = ckpt_data["stage"]
        lightning_module.stage = stage

    devices = None

    pl.callbacks.ModelCheckpoint.CHECKPOINT_NAME_LAST = f"chimp_{model_name}"
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_path,
            filename=f"chimp_{model_name}",
            verbose=True,
            save_top_k=0,
            save_last=True,
        ),
    ]

    all_optimizers = []
    all_schedulers = []
    all_callbacks = []
    opt_prev = None
    for stage_ind, training_config in enumerate(training_configs):
        opt_s, sch_s, cback_s = get_optimizer_and_scheduler(
            training_config, mrnn.model, previous_optimizer=opt_prev
        )
        opt_prev = opt_s
        all_optimizers.append(opt_s)
        all_schedulers.append(sch_s)
        all_callbacks.append(cback_s)

    lightning_module.optimizer = all_optimizers
    lightning_module.scheduler = all_schedulers

    for stage_ind, training_config in enumerate(training_configs):
        if stage_ind < lightning_module.stage:
            continue

        # Restore LR if optimizer is reused.
        if training_config.reuse_optimizer:
            if "lr" in training_config.optimizer_args:
                optim = lightning_module.optimizer[stage_ind]
                lr = training_config.optimizer_args["lr"]
                for group in optim.param_groups:
                    group["lr"] = lr

        stage_callbacks = callbacks + all_callbacks[stage_ind]
        training_loader, validation_loader = create_data_loaders(
            mrnn.model_config, training_config, training_data_path, validation_data_path
        )

        devices = training_config.devices
        if devices is None:
            if training_config.accelerator in ["cuda", "gpu"]:
                devices = -1
            else:
                devices = 1
        lightning_module.stage_name = training_config.name

        trainer = pl.Trainer(
            default_root_dir=output_path,
            max_epochs=training_config.n_epochs,
            accelerator=training_config.accelerator,
            devices=devices,
            precision=training_config.precision,
            logger=lightning_module.tensorboard,
            callbacks=stage_callbacks,
            num_sanity_val_steps=0,
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
            enable_progress_bar=True,
        )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=ckpt_path,
        )
        mrnn.save(output_path / f"chimp_{model_name}.pckl")
        ckpt_path = None


def find_most_recent_checkpoint(path: Path, model_name: str) -> Path:
    """
    Find most recente Pytorch lightning checkpoint files.

    Args:
        path: A pathlib.Path object pointing to the folder containing the
            checkpoints.
        model_name: The model name as defined by the user.

    Return:
        If a checkpoint was found, returns a object pointing to the
        checkpoint file with the highest version number. Otherwise
        returns 'None'.
    """
    path = Path(path)

    checkpoint_files = list(path.glob(f"chimp_{model_name}*.ckpt"))
    if len(checkpoint_files) == 0:
        return None
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    checkpoint_regexp = re.compile(rf"chimp_{model_name}(-v\d*)?.ckpt")
    versions = []
    for checkpoint_file in checkpoint_files:
        match = checkpoint_regexp.match(checkpoint_file.name)
        if match is None:
            return None
        if match.group(1) is None:
            versions.append(-1)
        else:
            versions.append(int(match.group(1)[2:]))
    ind = np.argmax(versions)
    return checkpoint_files[ind]


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
    gradient_clipping: Optional[float] = None
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False
    metrics: Optional[Dict[str, List["Metric"]]] = None
    include_input_steps: bool = False
    require_input: bool = False

    log_every_n_steps: Optional[int] = None
    gradient_clip_val: Optional[float] = None
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
            "sample_rate", int, config_dict, f"training stage '{name}'", 1
        )
        sequence_length = get_config_attr(
            "sequence_length", int, config_dict, f"training stage '{name}'", 1
        )
        forecast = get_config_attr(
            "forecast", int, config_dict, f"training stage '{name}'", 0
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
            "scheduler", str, config_dict, f"training stage '{name}'", "none"
        )
        if scheduler == "none":
            scheduler = None
        scheduler_args = get_config_attr(
            "scheduler_args", dict, config_dict, f"training stage '{name}'", {}
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
            "include_input_steps", bool, config_dict, f"training stage '{name}'", False
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
        accumulate_grad_batches = get_config_attr(
            "accumulate_grad_batches", int, config_dict, f"training stage {name}", None
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
            shrink_output=shrink_output,
            scene_size=scene_size,
            augment=augment,
            time_step=time_step,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
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
                missing_value_policy="none",
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
                shrink_output=self.shrink_output,
                missing_value_policy="none",
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
                missing_value_policy="none",
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
                shrink_output=self.shrink_output,
                missing_value_policy="none",
                augment=False,
                validation=True,
                include_input_steps=self.include_input_steps,
                require_input=self.require_input
            )


@click.argument("experiment_name")
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
    experiment_name: str,
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    resume: bool = False,
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

    module = LightningRetrieval(retrieval_model, "retrieval_module", training_schedule)

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
