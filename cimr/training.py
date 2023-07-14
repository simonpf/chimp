"""
cimr.training
=============

Module implementing training functionality.
"""
from pathlib import Path

import torch
from torch import nn
import pytorch_lightning as pl

from quantnn import metrics
from torch.utils.data import DataLoader

from cimr.data.training_data import CIMRDataset


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
        num_workers=training_config.data_loader_workers
    )
    print("NUM LOADERS :: ", training_config.data_loader_workers)
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
        batch_size=8 * training_config.batch_size,
        num_workers=training_config.data_loader_workers
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
        metrics.ScatterPlot(log_scale=True),
        metrics.CalibrationPlot()
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
        optimizer, scheduler = training_config.get_optimizer_and_scheduler(
            mrnn.model
        )
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
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
        )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
        )

        mrnn.save(model_path / f"cimr_{model_name}.pckl")
