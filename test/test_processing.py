"""
Tests for the chimp.processing module.
"""
from pathlib import Path

import numpy as np
from quantnn.qrnn import QRNN
import torch

from chimp.data.training_data import SingleStepDataset
from chimp.processing import retrieval_step, empty_input

from conftest import (
    mrms_surface_precip_data,
    cpcir_data,
    gmi_data,
    cpcir_gmi_mrnn
)

def test_retrieval_step(
        mrms_surface_precip_data,
        cpcir_data,
        gmi_data,
        cpcir_gmi_mrnn
):
    """
    Test performing the retrieval over multiple time steps.
    """
    data_path = mrms_surface_precip_data
    model = cpcir_gmi_mrnn
    input_data = SingleStepDataset(
        data_path,
        inputs=["cpcir", "gmi"],
        reference_data="mrms",
        missing_value_policy="random"
    )
    data_iterator = input_data.full_domain(
        start_time=np.datetime64("2020-01-01T04:00:00"),
        end_time=np.datetime64("2020-01-01T06:10:00"),
    )
    for time, x, y in data_iterator:
        retrieval_step(
            model,
            x,
            None,
            device="cpu",
            float_type=torch.bfloat16
        )
