"""
chimp.processing
===============

Routines for the processing of retrievals and forecasts.
"""
import logging
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from torch import nn
import xarray as xr
from pansat.time import to_datetime
from pytorch_retrieve.architectures import load_model

from chimp.tiling import Tiler
from chimp.data.input import InputLoader


LOGGER = logging.getLogger(__name__)



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
    model, model_input,  tile_size=256, device="cuda", float_type=torch.float32
):
    """
    Run retrieval on given input.

    Args:
        model: The CHIMP model to perform the retrieval with.
        model_input: A dict containing the input for the given time step.
        tile_size: The size to use for the tiling.

    Return:
        An ``xarray.Dataset`` containing the retrieval results.
    """
    x = model_input
    x = {name: tensor[None].to(dtype=float_type, device=device) for name, tensor in x.items()}
    tiler = Tiler(x, tile_size=tile_size, overlap=32)

    means = {}

    model = model.to(device=device, dtype=float_type).eval()

    def predict_fun(x_t):
        results = {}

        with torch.no_grad():
            if device != "cpu":
                with torch.autocast(device_type=device, dtype=float_type):
                    y_pred = model(x_t)
                    for key, y_pred_k in y_pred.items():
                        y_mean_k = y_pred_k.expected_value()[0, 0]
                        results[key + "_mean"] = y_mean_k.cpu().numpy()
            else:
                y_pred = model(x_t)
                for key, y_pred_k in y_pred.items():
                    y_mean_k = y_pred_k.expected_value()[0, 0]
                    results[key + "_mean"] = y_mean_k.cpu().numpy()
        return results

    dims = ("classes", "y", "x")
    results = tiler.predict(predict_fun)
    results = xr.Dataset(
        {key: (dims[-value.ndim :], value) for key, value in results.items()}
    )
    return results


@click.argument("model")
@click.argument("input_datasets", nargs=-1)
@click.argument("input_path")
@click.argument("output_path")
@click.option("--device", type=str, default="cuda")
@click.option("--precision", type=str, default="single")
@click.option("--tile_size", type=int, default=128)
@click.option("-v", "--verbose", count=True)
def cli(
        model: Path,
        input_datasets: List[str],
        input_path: Path,
        output_path: Path,
        device: str = "cuda",
        precision: str = "single",
        tile_size: int = 128,
        verbose: int = 0
) -> int:
    """
    Process input files.
    """
    input_data = InputLoader(input_path, input_datasets)
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

    for time, model_input in input_data:
        LOGGER.info("Starting processing input @ %s", time)
        results = retrieval_step(
            model,
            model_input,
            tile_size=tile_size,
            device=device,
            float_type=float_type
        )
        LOGGER.info("Finished processing input @ %s", time)

        results["time"] = time.astype("datetime64[ns]")
        date = to_datetime(time)
        date_str = date.strftime("%Y%m%d_%H_%M")
        output_file = output_path / f"chimp_{date_str}.nc"
        LOGGER.info("Writing results to %s", output_file)
        results.to_netcdf(output_path / f"chimp_{date_str}.nc")
