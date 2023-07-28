"""
cimr.training
=============

Module implementing training functionality.
"""
from pathlib import Path
import re

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from pytorch_lightning.callbacks import Callback
from quantnn import metrics
from torch.utils.data import DataLoader

from cimr.data.training_data import CIMRDataset


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
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        mrnn.model.apply(reset_params)


def get_optimizer_and_scheduler(training_config, model):
    """
    Return torch optimizer and and learning-rate scheduler objects
    corresponding to this configuration.

    Args:
        model: The model to be trained as a torch.nn.Module object.
    """
    optimizer = getattr(torch.optim, training_config.optimizer)
    optimizer = optimizer(model.parameters(), **training_config.optimizer_kwargs)

    scheduler = training_config.scheduler
    if scheduler is None:
        return optimizer, None, []

    if scheduler == "lr_search":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=2.51
        )
        callbacks = [
            ResetParameters()
        ]
        return optimizer, scheduler, callbacks

    scheduler = getattr(torch.optim.lr_scheduler, training_config.scheduler)
    scheduler = scheduler(
        optimizer=optimizer,
        **training_config.scheduler_kwargs,
    )
    return optimizer, scheduler, []


def create_data_loaders(
        model_config,
        training_config,
        training_data_path,
        validation_data_path
):

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
                    "Retrieval outputs must all come from the same "
                    " reference data."
                )

    training_data = CIMRDataset(
        training_data_path,
        inputs=inputs,
        reference_data=reference_data,
        sample_rate=training_config.sample_rate,
        sequence_length=training_config.sequence_length,
        window_size=training_config.input_size,
        quality_threshold=training_config.quality_threshold
    )
    training_loader = DataLoader(
        training_data,
        shuffle=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=training_data.init_rng,
        pin_memory=True
    )
    if validation_data_path is None:
        return training_loader, None

    validation_data =  CIMRDataset(
        validation_data_path,
        inputs=inputs,
        reference_data=reference_data,
        sample_rate=training_config.sample_rate,
        sequence_length=training_config.sequence_length,
        window_size=training_config.input_size,
        quality_threshold=training_config.quality_threshold
    )
    validation_loader = DataLoader(
        validation_data,
        shuffle=False,
        batch_size=2 * training_config.batch_size,
        num_workers=training_config.data_loader_workers,
        worker_init_fn=validation_data.init_rng,
        pin_memory=True
    )
    return training_loader, validation_loader



def train(
        model_name,
        mrnn,
        training_configs,
        training_data_path,
        validation_data_path,
        output_path,
        ckpt_path=None
):
    """
    Train a CIMR retrieval model using Pytorch lightning.

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
    model_path = output_path / model_name
    model_path.mkdir(exist_ok=True, parents=True)

    mtrcs = [
        metrics.Bias(),
        metrics.Correlation(),
        metrics.CRPS(),
        metrics.MeanSquaredError(),
        #metrics.ScatterPlot(log_scale=True),
        #metrics.CalibrationPlot()
    ]

    lightning_module = mrnn.lightning(
        mask=-100,
        metrics=mtrcs,
        name=model_name
    )

    devices = None

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=model_path,
            filename=f"cimr_{model_name}"
        )
    ]

    for training_config in training_configs:
        optimizer, scheduler, clbks = get_optimizer_and_scheduler(
            training_config,
            mrnn.model
        )
        callbacks = callbacks + clbks
        training_loader, validation_loader = create_data_loaders(
            mrnn.model_config,
            training_config,
            training_data_path,
            validation_data_path
        )

        if training_config.accelerator in ["cuda", "gpu"]:
            devices = -1
        else:
            devices = 8

        lightning_module.optimizer = optimizer
        lightning_module.scheduler = scheduler

        trainer = pl.Trainer(
            default_root_dir=model_path,
            max_epochs=training_config.n_epochs,
            accelerator=training_config.accelerator,
            devices=devices,
            precision=training_config.precision,
            logger=lightning_module.tensorboard,
            callbacks=callbacks,
            #strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=ckpt_path
        )

        mrnn.save(model_path / f"cimr_{model_name}.pckl")


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

    checkpoint_files = list(path.glob("*.ckpt"))
    print("CKPT :: ", path, checkpoint_files)
    if len(checkpoint_files) == 0:
        return None
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    checkpoint_regexp = re.compile(f"cimr_{model_name}(-v\d*)?.ckpt")
    versions = []
    for checkpoint_file in checkpoint_files:
        match = checkpoint_regexp.match(checkpoint_file.name)
        if match.group(1) is None:
            versions.append(-1)
        else:
            versions.append(int(match.group(1)[2:]))
    ind = np.argmax(versions)
    return checkpoint_files[ind]
