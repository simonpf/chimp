"""
Test the training of CIMR retrieval models.
"""
from pathlib import Path
import os

import numpy as np
import pytest

from cimr import models
from cimr.config import TrainingConfig
from cimr.data import inputs, reference
from cimr.data.training_data import CIMRDataset
from cimr.models import compile_mrnn
from cimr.training import (
    train,
    find_most_recent_checkpoint
)

from torch.utils.data import DataLoader

TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

@NEEDS_TEST_DATA
def test_training(tmp_path):

    model_config = models.load_config("gremlin")
    model_config.input_configs = [
        models.InputConfig(
            inputs.CPCIR,
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

    training_data = CIMRDataset(
        TEST_DATA / "training_data",
        reference_data="mrms",
        inputs=["cpcir"],
    )
    training_loader = DataLoader(
        training_data,
        shuffle=True
    )
    validation_loader = DataLoader(
        training_data,
        shuffle=False
    )

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
            sample_rate=4
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
            sample_rate=4
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
            sample_rate=4,
            reuse_optimizer=True
        )
    ]

    mrnn = compile_mrnn(model_config)

    train(
        "test_model",
        mrnn,
        training_configs,
        TEST_DATA / "training_data",
        TEST_DATA / "training_data",
        tmp_path,
    )

    ckpt_path = find_most_recent_checkpoint(tmp_path, "test_model")
    print("CKPT PATH :: ", ckpt_path)

    train(
        "test_model",
        mrnn,
        training_configs,
        TEST_DATA / "training_data",
        TEST_DATA / "training_data",
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
