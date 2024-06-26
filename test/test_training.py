"""
Test the training of CHIMP retrieval models.
"""
from pathlib import Path
import os

import numpy as np
import pytest

from conftest import (
    mrms_surface_precip_data,
    cpcir_data,
    gmi_data
)

from chimp import models
from chimp.config import TrainingConfig
from chimp.data import get_input_dataset, get_reference_dataset
from chimp.training import (
    train,
    find_most_recent_checkpoint
)

import torch
from torch.utils.data import DataLoader


def test_training(
        tmp_path,
        mrms_surface_precip_data,
        cpcir_data
):
    """
    Tests training over multiple epochs and the termination when
    the learning rates falls below the minimum LR.
    """
    model_config = models.load_config("gremlin")
    model_config.input_configs = [
        models.InputConfig(
            get_input_dataset("cpcir"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
        models.InputConfig(
            get_input_dataset("gmi"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    model_config.output_configs = [
        models.OutputConfig(
            get_reference_dataset("mrms"),
            "surface_precip",
            "mse",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    model = models.compile_model(model_config)

    acc = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "16" if torch.cuda.is_available() else "16"
    sample_rate = 4 if torch.cuda.is_available() else 1

    training_configs = [
        TrainingConfig(
            "Stage 1",
            4,
            "SGD",
            {"lr": 1e-4},
            scheduler = "ReduceLROnPlateau",
            scheduler_kwargs = {"patience": 1, "min_lr": 1e-3},
            minimum_lr = 1e-2,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            reuse_optimizer=True,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        )
    ]

    mrnn = compile_mrnn(model_config)


    train(
        "test_model",
        mrnn,
        training_configs,
        cpcir_data,
        cpcir_data,
        tmp_path,
    )

    ckpt_path = find_most_recent_checkpoint(tmp_path, "test_model")

    train(
        "test_model",
        mrnn,
        training_configs,
        tmp_path,
        tmp_path,
        tmp_path,
        ckpt_path=ckpt_path
    )


def test_training_multi_input(
        tmp_path,
        mrms_surface_precip_data,
        cpcir_data
):
    """
    Tests training over multiple epochs and the termination when
    the learning rates falls below the minimum LR.
    """
    model_config = models.load_config("gremlin")
    model_config.input_configs = [
        models.InputConfig(
            get_input_dataset("cpcir"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
        models.InputConfig(
            get_input_dataset("gmi"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    model_config.output_configs = [
        models.OutputConfig(
            get_reference_dataset("mrms"),
            "surface_precip",
            "mse",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    model = models.compile_model(model_config)

    acc = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "16" if torch.cuda.is_available() else "16"
    sample_rate = 4 if torch.cuda.is_available() else 1

    training_configs = [
        TrainingConfig(
            "Stage 1",
            4,
            "SGD",
            {"lr": 1e-4},
            scheduler = "ReduceLROnPlateau",
            scheduler_kwargs = {"patience": 1, "min_lr": 1e-3},
            minimum_lr = 1e-2,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            reuse_optimizer=True,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing"
        )
    ]

    mrnn = compile_mrnn(model_config)


    train(
        "test_model",
        mrnn,
        training_configs,
        cpcir_data,
        cpcir_data,
        tmp_path,
    )

    ckpt_path = find_most_recent_checkpoint(tmp_path, "test_model")

    train(
        "test_model",
        mrnn,
        training_configs,
        tmp_path,
        tmp_path,
        tmp_path,
        ckpt_path=ckpt_path
    )

def test_training_masked_input(
        tmp_path,
        mrms_surface_precip_data,
        cpcir_data
):
    """
    Tests training over multiple epochs and the termination when
    the learning rates falls below the minimum LR.
    """
    model_config = models.load_config("gremlin")
    model_config.input_configs = [
        models.InputConfig(
            get_input_dataset("cpcir"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1,
            deep_supervision=True
        ),
        models.InputConfig(
            get_input_dataset("gmi"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1,
            deep_supervision=True
        ),
        models.InputConfig(
            get_input_dataset("mhs"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1,
            deep_supervision=True
        ),
    ]
    model_config.output_configs = [
        models.OutputConfig(
            get_reference_dataset("mrms"),
            "surface_precip",
            "quantile_loss",
            quantiles=np.linspace(0, 1, 34)[1:-1]
        ),
    ]

    model = models.compile_model(model_config)

    acc = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "16" if torch.cuda.is_available() else "16"
    sample_rate = 4 if torch.cuda.is_available() else 1

    training_configs = [
        TrainingConfig(
            "Stage 1",
            4,
            "SGD",
            {"lr": 1e-4},
            scheduler = "ReduceLROnPlateau",
            scheduler_kwargs = {"patience": 1, "min_lr": 1e-3},
            minimum_lr = 1e-2,
            batch_size=2,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="masked"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="masked"
        ),
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            reuse_optimizer=True,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="masked"
        )
    ]

    mrnn = compile_mrnn(model_config)


    train(
        "test_model",
        mrnn,
        training_configs,
        cpcir_data,
        cpcir_data,
        tmp_path,
    )

    ckpt_path = find_most_recent_checkpoint(tmp_path, "test_model")

    train(
        "test_model",
        mrnn,
        training_configs,
        tmp_path,
        tmp_path,
        tmp_path,
        ckpt_path=ckpt_path
    )

def test_find_most_recent_checkpoint(tmp_path):
    """
    Test that the find_most_recent_checkpoint function identifies the
    correct checkpoint file.
    """
    ckpt_1 = tmp_path / ("chimp_model.ckpt")
    ckpt_2 = tmp_path / ("chimp_model-v1.ckpt")
    ckpt_3 = tmp_path / ("chimp_model-v12.ckpt")
    model_name = "model"

    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt is None

    ckpt_1.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_1

    ckpt_2.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_2

    ckpt_3.touch()
    ckpt = find_most_recent_checkpoint(tmp_path, model_name)
    assert ckpt == ckpt_3


def test_sequence_training(
        tmp_path,
        mrms_surface_precip_data,
        gmi_data,
        cpcir_data
):
    """
    Tests training over multiple epochs and the termination when
    the learning rates falls below the minimum LR.
    """
    model_config = models.load_config("gremlin")
    model_config.input_configs = [
        models.InputConfig(
            get_input_dataset("cpcir"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
        models.InputConfig(
            get_input_dataset("gmi"),
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    model_config.output_configs = [
        models.OutputConfig(
            get_reference_dataset("mrms"),
            "surface_precip",
            "mse",
            temporal_merging=True
        ),
    ]

    model = models.compile_model(model_config)

    acc = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "16" if torch.cuda.is_available() else "16"
    sample_rate = 4 if torch.cuda.is_available() else 1

    training_configs = [
        TrainingConfig(
            "Stage 2",
            4,
            "SGD",
            {"lr": 1e-2},
            scheduler="CosineAnnealingLR",
            scheduler_kwargs={
                "T_max": 4,
                "verbose": True
            },
            minimum_lr=1e-4,
            batch_size=1,
            sample_rate=sample_rate,
            accelerator=acc,
            precision=prec,
            data_loader_workers=1,
            missing_value_policy="missing",
            sequence_length=4
        )
    ]

    mrnn = compile_mrnn(model_config)


    train(
        "test_model",
        mrnn,
        training_configs,
        cpcir_data,
        cpcir_data,
        tmp_path,
    )
