"""
chimp.forecast
==============

Functionality for performing forecasts with CHIMP models.
"""
from typing import Optional, Union

import numpy as np
import torch
import xarray as xr


def tensor_to(tensor, device, dtype=None):
    if isinstance(tensor, list):
        return [tensor_to(t, device, dtype=dtype) for t in tensor]
    if isinstance(tensor, tuple):
        return tuple([tensor_to(t, device, dtype=dtype) for t in tensor])
    elif isinstance(tensor, dict):
        return {name: t for name, t in tensor.items()}
    return tensor.to(device, dtype=dtype)



def run_forecast(
        data_loader,
        model,
        initialization_time: np.datetime64,
        input_steps: int,
        forecast_steps: int,
        device: Optional[Union[str, torch.device]] = None
):
    """
    Run forecast using a CHIMP forecast model.

    Args:
        data_laoder: A data loader providing access to the input data.
        model: The CHIMP forecast model.
        input_steps: The number of model time steps to use as input.
        forecast_steps: The numebr of forecasts steps to perform.
        device: The device on which to perform the forecast.

    Return:
        An xarray.Dataset containing the forecast results.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    time_step = np.timedelta64(model.time_step, "m")
    input_start_time = initialization_time - time_step * input_steps

    input_data = data_loader.get_input_dataset(input_start_time)
    lead_time = torch.tensor([np.arange(1, forecast_steps + 1) * model.time_step])
    input_data["lead_time"] = lead_time
    input_data = {
        name: tensor_to(tensor, device) for name, tensor in input_data.items()
    }

    model = model.eval().to(device=device)

    results = {}

    with torch.no_grad():
        pred = model(input_data)
        for name, tensors in pred.items():
            seq = []
            for tensor in tensors:
                seq.append(tensor.expected_value().cpu().numpy()[0, 0])
            results[name] = (("step", "y", "x"), np.stack(seq, 0))

    results = xr.Dataset(results)
    steps = (np.arange(0, forecast_steps) * model.time_step).astype("timedelta64[m]")
    results["step"] = steps
    return xr.Dataset(results)
