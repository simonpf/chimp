"""
==============
cimr.bin.train
==============

This sub-module implements the cimr CLI to train the retrieval.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
    is called from the top-level parser defined in 'cimr.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Train retrieval network.",
        description=(
            """
            Train the retrieval network.
            """
        ),
    )
    parser.add_argument(
        "training_data",
        metavar="training_data",
        type=str,
        help="Folder containing the training data.",
    )
    parser.add_argument(
        "model_path",
        metavar="model_path",
        type=str,
        help="Path to store the trained model.",
    )
    parser.add_argument(
        "--validation_data",
        metavar="path",
        type=str,
        help="Folder containing the validation data.",
        default=None
    )
    parser.add_argument(
        "--n_stages",
        metavar="N",
        type=int,
        default=4,
        help="Number of stages in encoder/decoder architecture.",
    )
    parser.add_argument(
        "--n_features",
        metavar="N",
        type=int,
        default=128,
        help="Number of features in first encoder stage.",
    )
    parser.add_argument(
        "--n_blocks",
        metavar="N",
        type=int,
        default=4,
        help="Number of blocks per encoder stage.",
    )
    parser.add_argument(
        "--batch_size",
        metavar="N",
        type=int,
        default=4,
        help="The batch size to use during training.",
    )
    parser.add_argument(
        "--lr",
        metavar="lr",
        type=float,
        default=0.0005,
        help="The learning rate with which to start the training",
    )
    parser.add_argument(
        "--n_epochs",
        metavar="n_epochs",
        type=int,
        default=20,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--accelerator",
        metavar="device",
        type=str,
        default="gpu",
        help="The accelerator to use for training.",
    )
    parser.add_argument(
        "--sequence_length",
        metavar="n",
        type=int,
        default=1,
        help="Length of training sequences."
    )
    parser.add_argument(
        "--forecast",
        metavar="n",
        type=int,
        default=None,
        help="Number of forecast steps to train the model on."
    )
    parser.add_argument(
        "--input_size",
        metavar="n",
        type=int,
        default=256,
        help="Size of the input scenes at 2 km resolution."
    )
    parser.add_argument(
        "--quality_threshold",
        metavar="q",
        type=float,
        default=0.8,
        help="Quality index threshold for radar data."
    )
    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        default=None,
        help="Name to use for logging."
    )
    parser.add_argument(
        "--freeze_norms",
        action="store_true",
        help="Use running statistics in normalization layers."
    )
    parser.add_argument(
        "--freeze_obs_branch",
        action="store_true",
        help="Train only temporal stepping and merging modules."
    )
    parser.add_argument(
        "--gradient_clipping",
        default=None,
        type=float,
        help="Threshold value for gradient clipping."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=["cpcir", "gmi"]
    )
    parser.add_argument(
        "--reference_data",
        type=str,
        default="mrms"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=16
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None
    )
    parser.add_argument("--pretrain", action="store_true")

    parser.set_defaults(func=run)


def run(args):
    """
    Run training.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from torch.optim import AdamW, SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, Callback
    from pytorch_lightning.strategies import DDPStrategy
    from quantnn.qrnn import QRNN
    from quantnn import metrics
    from quantnn import transformations
    from cimr import models

    #
    # Prepare training and validation data.
    #

    from cimr.data.training_data import (CIMRSequenceDataset,
                                         CIMRDataset,
                                         CIMRPretrainDataset,
                                         sparse_collate)

    from torch.utils.data import DataLoader

    training_data = Path(args.training_data)
    if not training_data.exists():
        LOGGER.error(
            "Provided training data path '%s' doesn't exist."
        )
        sys.exit()

    if args.sequence_length == 1:
        training_data = CIMRDataset(
            training_data,
            window_size=args.input_size,
            quality_threshold=args.quality_threshold,
            inputs=args.inputs,
            reference_data=args.reference_data,
            sample_rate=args.sample_rate
        )
    elif args.pretrain:
        training_data = CIMRPretrainDataset(
            training_data,
            window_size=args.input_size,
            quality_threshold=args.quality_threshold,
            inputs=args.inputs,
            reference_data=args.reference_data,
            sample_rate=args.sample_rate
        )
    else:
        training_data = CIMRSequenceDataset(
            training_data,
            sequence_length=args.sequence_length,
            window_size=args.input_size,
            quality_threshold=args.quality_threshold,
            forecast=args.forecast,
            inputs=args.inputs,
            sample_rate=args.sample_rate
        )

    training_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        num_workers=16,
        worker_init_fn=training_data.init_rng,
        collate_fn=sparse_collate,
        shuffle=True,
        pin_memory=True
    )

    validation_data = args.validation_data
    validation_loader = None
    if validation_data is not None:
        validation_data = Path(validation_data)
        if not validation_data.exists():
            LOGGER.error(
                "Provided validation data path '%s' doesn't exist."
            )
            sys.exit()
        if args.sequence_length == 1:
            validation_data = CIMRDataset(
                validation_data,
                window_size=args.input_size,
                quality_threshold=args.quality_threshold,
                reference_data=args.reference_data,
                inputs=args.inputs
            )
        elif args.pretrain:
            validation_data = CIMRPretrainDataset(
                validation_data,
                window_size=args.input_size,
                quality_threshold=args.quality_threshold,
                reference_data=args.reference_data,
                inputs=args.inputs
            )
        else:
            validation_data = CIMRSequenceDataset(
                validation_data,
                sequence_length=args.sequence_length,
                window_size=args.input_size,
                quality_threshold=args.quality_threshold
            )

        validation_loader = DataLoader(
            validation_data,
            batch_size=4 * args.batch_size,
            num_workers=8,
            worker_init_fn=validation_data.init_rng,
            shuffle=False,
            collate_fn=sparse_collate,
            pin_memory=True
        )

    #
    # Create model
    #

    n_stages = args.n_stages
    n_blocks = args.n_blocks
    n_features = args.n_features

    model_path = Path(args.model_path)


    if model_path.exists() and not model_path.is_dir():
        qrnn = QRNN.load(model_path)
        model = qrnn.model
    else:
        if args.model_type is None:
            if args.sequence_length == 1:
                model_type = "CIMRBaselineV3"
            else:
                model_type = "CIMRSeq"
        else:
            model_type = args.model_type
        model_type = getattr(models, model_type)
        model = model_type(
            n_stages=args.n_stages,
            n_blocks=args.n_blocks,
            inputs=args.inputs,
            reference_data=args.reference_data
        )

        quantiles = np.linspace(0, 1, 34)[1:-1]
        qrnn = QRNN(
            model=model,
            quantiles=quantiles,
            transformation=transformations.LogLinear()
        )

    #
    # Run training
    #

    metrics = [
        metrics.Bias(),
        metrics.Correlation(),
        metrics.CRPS(),
        metrics.MeanSquaredError(),
        metrics.ScatterPlot(log_scale=True),
        metrics.CalibrationPlot()
    ]
    lm = qrnn.lightning(mask=-100, metrics=metrics, name=args.name)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    lm.optimizer = optimizer
    lm.scheduler = scheduler

    devices = None
    if args.accelerator in ["cuda", "gpu"]:
        devices = -1
    else:
        devices = 8


    class FreezeObsBranch(Callback):
        """
        Callback class to freeze norm layers.
        """
        def on_train_epoch_start(self, trainer, pl_module):
            pl_module.model.encoder.train(False)
            pl_module.model.decoder.train(False)
            pl_module.model.head.train(False)

    callbacks = [LearningRateMonitor()]
    if args.freeze_norms:
        callbacks.append(FreezeNorms())
    if args.freeze_obs_branch:
        lm.model.encoder.freeze()
        lm.model.decoder.freeze()
        lm.model.head.freeze()
        callbacks.append(FreezeObsBranch())

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        logger=lm.tensorboard,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        #replace_sampler_ddp=True,
        gradient_clip_val=args.gradient_clipping,
        enable_checkpointing=False,
    )
    trainer.fit(
        model=lm,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )
    qrnn.save(model_path)
