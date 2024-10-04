"""
chimp.processing
===============

Routines for the processing of retrievals and forecasts.
"""
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import click
import numpy as np
import torch
from torch import nn
import xarray as xr
from pansat.time import to_datetime
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.modules.output import Quantiles

from chimp.tiling import Tiler
from chimp.data.input import (
    InputLoader,
    SequenceInputLoader,
    get_input_map,
)



LOGGER = logging.getLogger(__name__)



def iter_tensors(list_or_tensor: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Iterate over a single tensor or list of tensors.

    Args:
        list_or_tensor: A single tensor or list of tensors to iterate over.

    Return:
        A iterator over the tensors provided in 'list_or_tensor'.
    """
    if isinstance(list_or_tensor, torch.Tensor):
        yield list_or_tensor
    else:
        for tensor in list_or_tensor:
            yield tensor

def apply(fun, tensors):
    """
    Recursively apply function to collection of tensors.

    Args:
        fun: A callable taking and modifying a single tensor.
        tensors: A single tensor or an arbitrarily-nested list,
        tuple, or dict of tensors.

    Return:
        The same collection of tensor with 'fun' applied to each tensor in it.
    """
    if isinstance(tensors, torch.Tensor):
        return fun(tensors)
    if isinstance(tensors, list):
        return [apply(fun, tensor) for tensor in tensors]
    if isinstance(tensors, tuple):
        return tuple([apply(fun, tensor) for tensor in tensors])
    if isinstance(tensors, dict):
        return {
            name: apply(fun, tensor) for name, tensor in tensors.items()
        }
    raise ValueError(
        "Encountered unsupported input of type {type(tensors)} in 'apply'."
    )


    
def empty_input(model, model_input):
    """
    Determines whether the lacks the observations required by the
    given model.

    Args:
        model: The CHIMP model it use for the retrieval.
        model_input: A dict containing the input for a given time step.

    Return:
        ``True`` if the required inputs are missing, ``False`` otherwise.
    """
    empty = True
    for source in model.model.sources:
        if source == "mw":
            keys = ["mw_90", "mw_160", "mw_183"]
            empty = empty and all([not not_empty(model_input[mw]) for mw in keys])
        else:
            empty = empty and not not_empty(model_input[source])
    return empty


def retrieval_step(
        model,
        model_input: Dict[str, torch.Tensor],
        tile_size: int = 256,
        spatial_overlap: Optional[int] = None,
        device: str = "cuda",
        float_type: torch.dtype = torch.float32
):
    """
    Run retrieval on given input.

    Args:
        model: The CHIMP model to perform the retrieval with.
        model_input: A dict containing the input for the given time step.
        tile_size: The size to use for the tiling.
        spatial_overlap: The number of overlap between neighboring tiles. Defaults
            1/4 of the tile size.
        sequence_length: Number of input steps expected by the model.
        forecast: The number of time steps to forecast.

    Return:
        An ``xarray.Dataset`` containing the retrieval results.
    """
    lead_time = model_input.pop("lead_time", None)
    input_map = get_input_map(model_input)
    if not isinstance(input_map, list):
        input_map = [input_map]

    input_names = list(model_input.keys())

    x = apply(
        lambda tensor: tensor.to(dtype=float_type, device=device),
        model_input
    )

    if spatial_overlap is None:
        spatial_overlap = tile_size // 4
    tiler = Tiler(x, tile_size=tile_size, overlap=spatial_overlap)

    means = {}

    device = torch.device(device)
    model = model.to(device=device, dtype=float_type).eval()

    quantile_outputs = {
        mod.name: mod.tau.cpu().numpy() for mod in model.modules()
        if isinstance(mod, Quantiles)
    }

    def predict_fun(x_t):
        results = {}

        if lead_time is not None:
            x_t["lead_time"] = lead_time

        with torch.no_grad():
            if device != "cpu":
                with torch.autocast(device_type="cuda", dtype=float_type):
                    y_pred = model(x_t)
                    for key, y_pred_k in y_pred.items():
                        for step, y_pred_k_s in enumerate(iter_tensors(y_pred_k)):
                            results_step = results.setdefault(step, {})
                            y_mean_k_s = y_pred_k_s.expected_value()[0, 0]
                            results_step[key + "_mean"] = y_mean_k_s.cpu().numpy()
                            if key in quantile_outputs:
                                results_step[key + "_cdf"] = y_pred_k_s.cpu().float().numpy()[0, :, 0]
            else:
                y_pred = model(x_t)
                for key, y_pred_k in y_pred.items():
                    for step, y_pred_k_s in enumerate(iter_tensors(y_pred_k)):
                        results_step = results.setdefault(step, {})
                        y_mean_k_s = y_pred_k_s.expected_value()[0, 0]
                        results_step[key + "_mean"] = y_mean_k.cpu().float().numpy()

                        if key in quantile_outputs:
                            results_step[key + "_cdf"] = y_pred_k_s.cpu().float().numpy()[0, :, 0]
        return results

    dims = ("tau", "y", "x")
    results = tiler.predict(predict_fun)

    if lead_time is not None:
        n_fc = lead_time.shape[1]
    else:
        n_fc = 0
    n_retrieved = len(results) - n_fc

    datasets = [None] * len(results)
    for step, results_s in results.items():
        results_s = xr.Dataset(
            {key: (dims[-value.ndim :], value) for key, value in results_s.items()}
        )
        tau = next(iter(quantile_outputs.values()))
        results_s["tau"] = ("tau", tau)

        if step < n_retrieved:
            results_s["input_map"] = (("inputs", "y", "x"), input_map[step].numpy()[0])
            results_s["inputs"] = (("inputs", input_names))
        datasets[step] = results_s

    return datasets


@click.argument("model")
@click.option("-i", "--inputs", "input_datasets", required=True)
@click.argument("input_paths", nargs=-1)
@click.argument("output_path")
@click.option("--device", type=str, default="cuda")
@click.option("--precision", type=str, default="single")
@click.option("--tile_size", type=int, default=128)
@click.option("--sequence_length", type=int, default=0)
@click.option("--temporal_overlap", type=int, default=None)
@click.option("--forecast", type=int, default=0)
@click.option("-v", "--verbose", count=True)
def cli(
        model: Path,
        input_datasets: List[str],
        input_paths: List[str],
        output_path: str,
        device: str = "cuda",
        precision: str = "single",
        tile_size: int = 128,
        sequence_length: int = 0,
        temporal_overlap: int = 8,
        forecast: int = 0,
        verbose: int = 0
) -> int:
    """
    Process input files.
    """
    inputs = [Path(path) for path in input_paths]
    if len(inputs) == 1:
        if inputs[0].exists() and inputs[0].is_file():
            input_path: Union[Path, List[Path]] = inputs
        elif inputs[0].exists() and inputs[0].is_dir():
            input_path = inputs[0]
        else:
            LOGGER.error(
                "Input '%s' does not exist or is not a file or directory.",
                input_paths[0],
            )
            return 1
    else:
        for input in inputs:
            if not input.exists() or not input.is_file():
                LOGGER.error("Input '%s' does not exist or is not a file.", input)
                return 1
        input_path = inputs

    input_datasets = input_datasets.split(",")
    if sequence_length < 2:
        input_data = InputLoader(input_path, input_datasets)
    else:
        if temporal_overlap is None:
            temporal_overlap = sequence_length - 1
        input_data = SequenceInputLoader(
            input_path,
            input_datasets,
            sequence_length=sequence_length,
            forecast=forecast,
            temporal_overlap=temporal_overlap
        )
    model = load_model(model)

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if verbose > 0:
        logging.basicConfig(level="INFO", force=True)

    if precision == "single":
        float_type = torch.float32
    else:
        float_type = torch.bfloat16

    for input_step, (time, model_input) in enumerate(input_data):
        LOGGER.info("Starting processing input @ %s", time)
        results = retrieval_step(
            model,
            model_input,
            tile_size=tile_size,
            device=device,
            float_type=float_type
        )
        LOGGER.info("Finished processing input @ %s", time)

        n_retrieved = len(results) - forecast

        curr_time = time
        drop_left = temporal_overlap // 2

        for step in range(n_retrieved):

            curr_time = time - (n_retrieved - step - 1) * input_data.time_step

            results_s = results[step]
            results_s["time"] = curr_time.astype("datetime64[ns]")
            date = to_datetime(curr_time)
            date_str = date.strftime("%Y%m%d_%H_%M")
            output_file = output_path / f"chimp_{date_str}.nc"

            if input_step > 0 and output_file.exists() and step < drop_left:
                continue

            LOGGER.info("Writing retrieval results to %s", output_file)
            results_s.to_netcdf(output_path / f"chimp_{date_str}.nc")

        if forecast > 0:
            results_forecast = results[n_retrieved:]
            lead_time = (np.arange(forecast) + 1.0) * input_data.time_step
            results = xr.concat(results_forecast, dim="step")
            results["lead_time"] = (("step",), lead_time.astype("timedelta64[ns]"))
            results["time"] = time.astype("datetime64[ns]")
            results["valid_time"] = (("step",), (time + lead_time).astype("datetime64[ns]"))

            date = to_datetime(time)
            date_str = date.strftime("%Y%m%d_%H_%M")
            output_file = output_path / f"chimp_forecast_{date_str}.nc"
            LOGGER.info("Writing forecast results to %s", output_file)
            results.to_netcdf(output_file)
