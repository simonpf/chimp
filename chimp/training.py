"""
chimp.training
=============

Module implementing training functionality.
"""
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from quantnn import metrics
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
            model.parameters(), **training_config.optimizer_kwargs
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
    scheduler_kwargs = training_config.scheduler_kwargs
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    scheduler = scheduler(
        optimizer=optimizer,
        **scheduler_kwargs,
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
            window_size=training_config.input_size,
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
            window_size=training_config.input_size,
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
            window_size=training_config.input_size,
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
            window_size=training_config.input_size,
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
            if "lr" in training_config.optimizer_kwargs:
                optim = lightning_module.optimizer[stage_ind]
                lr = training_config.optimizer_kwargs["lr"]
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
