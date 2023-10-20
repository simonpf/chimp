"""
Test the training of CIMR retrieval models.
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

from cimr import models
from cimr.config import TrainingConfig
from cimr.data import input, reference
from cimr.data.training_data import CIMRDataset
from cimr.models import compile_mrnn
from cimr.training import (
    train,
    find_most_recent_checkpoint
)

import torch
from torch.utils.data import DataLoader

TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

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
            input.CPCIR,
            stem_depth=1,
            stem_kernel_size=3,
            stem_downsampling=1
        ),
    ]
    model_config.output_configs = [
        models.OutputConfig(
            reference.MRMS,
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
    ckpt_1 = tmp_path / ("cimr_model.ckpt")
    ckpt_2 = tmp_path / ("cimr_model-v1.ckpt")
    ckpt_3 = tmp_path / ("cimr_model-v12.ckpt")
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
