"""
===============
chimp.bin.train
===============

This sub-module implements the chimp CLI to train the retrieval.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
    is called from the top-level parser defined in 'chimp.bin'.

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
        "experiment_name",
        metavar="name",
        type=str,
        help=("A name identifying the model to be trained."),
    )
    parser.add_argument(
        "--model_config",
        metavar="model_config",
        type=str,
        help=(
            "Path to a model configuration file "
            " or a model checkpoint. If not provided 'chimp' expects "
            " a file called 'model.toml' or 'model.yaml' file from which "
            " the configuration will be loaded."
        ),
        default=None,
    )
    parser.add_argument(
        "--training_config",
        metavar="model_config",
        type=str,
        help=(
            "Path to a training configuration file If not provided 'chimp' expects "
            " a file called 'training.toml' or 'training.yaml' in the current "
            " working directory  from which the configuration will be loaded."
        ),
        default=None,
    )
    parser.add_argument("--resume", action="store_true")

    parser.set_defaults(func=run)


def run(args):
    """
    Run training.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    import chimp.data.seviri
    import chimp.data.gpm
    import chimp.data.goes
    import chimp.data.cpcir
    import chimp.data.baltrad
    import chimp.data.mrms

    from pytorch_retrieve.architectures import load_and_compile_model
    from pytorch_retrieve.config import read_config_file, TrainingConfig
    from pytorch_retrieve.lightning import RetrievalModule
    from pytorch_retrieve.training import parse_training_config, run_training

    # Parse model config
    model_config = args.model_config
    if model_config is None:
        model_config = list(Path(".").glob("model.????"))
        if len(model_config) > 1:
            LOGGER.error(
                "No explicit path to model configuration file provided and "
                " the working directory contains more than one model.???? "
                "file."
            )
            return 1
        model_config = model_config[0]
        if not model_config.suffix in [".toml", ".yaml"]:
            LOGGER.error(
                "Model configuration file should be in '.toml' or '.yaml' " "format."
            )
    model = load_and_compile_model(model_config)

    # Parse training config
    training_config = args.training_config
    if training_config is None:
        training_config = list(Path(".").glob("training.????"))
        if len(training_config) == 0:
            LOGGER.error(
                "No explicit path to a training configuration file provided and "
                " the working directory does not contain a file "
                "matching the pattern training.????. "
            )
            return 1
        if len(training_config) > 1:
            LOGGER.error(
                "No explicit path to a training configuration file provided and "
                " the working directory contains more than one training.???? "
                "file."
            )
            return 1
        training_config = training_config[0]
        if not training_config.suffix in [".toml", ".yaml"]:
            LOGGER.error(
                "training configuration file should be in '.toml' or '.yaml' " "format."
            )

    training_schedule = parse_training_config(training_config)

    retrieval_module = RetrievalModule(model, training_schedule=training_schedule)
    run_training(Path("."), retrieval_module, None)
