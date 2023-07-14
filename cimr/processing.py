"""
cimr.processing
===============

Routines for the processing of retrievals and forecasts.
"""
import logging

import numpy as np
from quantnn.packed_tensor import PackedTensor
from quantnn.quantiles import (
    posterior_mean,
    sample_posterior,
    probability_larger_than
)
from quantnn.models.pytorch.lightning import to_device
import torch
from torch import nn
import xarray as xr

from cimr.models import not_empty
from cimr.tiling import Tiler


LOGGER = logging.getLogger(__file__)


def get_observation_mask(model_input, upsample=1):
    """
    Get a mask of valid observations.

    Args:
        x: A 4D ``torch.Tensor`` containing network of a given
            observation type.
        upsample: Factor by which the input should be upsampled.

    Return:
        A 3D, spatial mask that masks pixels that have valid observation
        input.
    """
    if isinstance(model_input, PackedTensor):
        model_input = model_input.tensor

    while upsample > 1:
        model_input = nn.functional.interpolate(
            model_input, scale_factor=2, mode="nearest"
        )
        upsample = upsample // 2

    return (model_input > -1.3).any(1)


def empty_input(model, model_input):
    """
    Determines whether the lacks the observations required by the
    given model.

    Args:
        model: The CIMR model it use for the retrieval.
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
        model_input,
        state,
        tile_size=256,
        device="cuda"
):
    """
    Run retrieval on given input.

    Args:
        model: The CIMR model to perform the retrieval with.
        model_input: A dict containing the input for the given time step.
        state: The current hidden state of the model.
        tile_size: The size to use for the tiling.

    Return:
        An ``xarray.Dataset`` containing the retrieval results and the
        crreutn internal state of the retrieval.
    """
    quantiles = torch.tensor(model.quantiles).to(device)
    model.model.to(device)

    x, y = model_input
    tiler = Tiler(x, tile_size=tile_size, overlap=64)

    means = {}

    for k, x_k in x.items():
        print(k, x_k.shape)


    for i in range(tiler.M):
        means.setdefault("surface_precip", []).append([])
        for j in range(tiler.N):

            x_t = to_device(tiler.get_tile(i, j), device)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.float16):
                    y_pred, state = model.model(x_t, state=state, return_state=True)
                    if isinstance(y_pred, PackedTensor):
                        y_pred = y_pred._t

                    for targ in ["surface_precip"]:
                        y_pred_t = model.transformation.invert(y_pred[targ])
                        means[targ][-1].append(
                            posterior_mean(
                                y_pred= y_pred_t,
                                quantile_axis=1,
                                quantiles=quantiles
                            ).cpu().numpy()[0]
                        )

    surface_precip = tiler.assemble(means["surface_precip"])

    results = xr.Dataset(
        {
            "surface_precip": (("y", "x"), surface_precip),
        }
    )
    return results, state


def make_forecast(
        qrnn,
        dataset,
        forecast_time,
        obs_steps,
        forecast_steps,
):
    """
    Make a precipitation forecasts for a given start time.

    Args:
        qrnn: The QRNN to use to make the forecast.
        dataset: The CIMRDataset providing the input data.
        forecast_time: The time for which to perform the forecast.
        obs_steps: Number of observations to ingest prior to making the
            forecast.
        forecast_steps: The number of steps to forecast.

    Return:
        A dataset containing the forecast.
    """
    state = None

    inputs = dataset.get_forecast_input(forecast_time, obs_steps)
    if inputs is None:
        return None

    LOGGER.info("Processing %s observations.", len(inputs))

    with torch.no_grad():
        for x, y, *_ in inputs:
            _, state = qrnn.model.forward(x, state=state, return_state=True)

    slice_y = inputs[0][2]
    slice_x = inputs[0][3]
    quantiles = torch.tensor(qrnn.quantiles)

    f_state = state
    y_pred_mean = []
    y_pred_sampled = []
    y_pred_prob = []

    LOGGER.info("Running forecast.")

    with torch.no_grad():

        for _ in range(forecast_steps):

            y_pred, f_state = qrnn.model(
                None, state=f_state, return_state=True
            )

            # Posterior mean.
            y_mean = posterior_mean(
                y_pred=y_pred,
                quantile_axis=1,
                quantiles=quantiles
            ).cpu().numpy()[0, slice_y, slice_x]
            y_pred_mean.append(y_mean)

            # P(dbz >= -5)
            prob = probability_larger_than(
                y_pred=y_pred,
                y=-5,
                quantile_axis=1,
                quantiles=quantiles
            ).cpu().numpy()[0, slice_y, slice_x]
            y_pred_prob.append(prob)

            # Sample from posterior
            sampled = sample_posterior(
                y_pred=y_pred,
                quantile_axis=1,
                quantiles=quantiles
            ).cpu().numpy()[0,  0, slice_y, slice_x]
            y_pred_sampled.append(sampled)

    y_pred_mean = np.stack(y_pred_mean)
    y_pred_sampled = np.stack(y_pred_sampled)
    y_pred_prob = np.stack(y_pred_prob)

    results = xr.Dataset({
        "dbz_mean": (("steps", "y", "x"), y_pred_mean),
        "dbz_prob": (("steps", "y", "x"), y_pred_prob),
        "dbz_sampled": (("steps", "y", "x"), y_pred_sampled)
    })
    return results
