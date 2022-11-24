"""
=============
cimr.bin.test
=============

This sub-module implements the cimr CLI to run CIMR models on
test data.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path
from queue import Queue

import numpy as np
import pandas as pd
import torch
from torch import nn
from quantnn import QRNN
from quantnn.packed_tensor import PackedTensor
from quantnn.quantiles import posterior_mean, sample_posterior, probability_larger_than
import xarray as xr

from cimr.models import not_empty
from cimr.processing import empty_input, retrieval_step

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'cimr.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "test",
        help="Run trained model on test data.",
        description=(
            """
            Run trained model on test data.
            """
        ),
    )
    parser.add_argument(
        "model",
        metavar="model",
        type=str,
        help="Path to the model to test.",
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=str,
        help="Path to the test data.",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Destination to store the test results.",
    )
    parser.add_argument(
        "--start_time",
        metavar="start",
        type=str,
        help="Optional start date to limit the testing period.",
    )
    parser.add_argument(
        "--end_time",
        metavar="end",
        type=str,
        help="Optional end date to limit the testing period.",
    )
    parser.add_argument(
        "--forecasts",
        metavar="n",
        type=int,
        help="The number of forecast steps to perform.",
        default=0,
    )
    parser.set_defaults(func=run)


def get_observation_mask(x, upsample=1):
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
    if isinstance(x, PackedTensor):
        x = x.tensor

    while upsample > 1:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        upsample = upsample // 2

    return (x > -1.3).any(1)


def make_forecasts(qrnn, state, quantiles, steps):
    """
    Make forecasts from a given hidden state.

    Args:
        qrnn: The QRNN instance to make the forecast with.
        state: The current hidden state.
        quantiles: A torch.Tensor containing the quantiles predicted with
            the QRNN.
        steps: The number of forecast steps to perform.

    Return:
        A tuple ``(y_pred, probs, y_pred_sampled)`` containing the predicted
        mean dBZ values in ``y_pred``, the estimated probability of observing
        a reflectivity higher than 0 in ``probs`` and independent samples from
        the posterior distribution in ``y_pred_sampled``.
    """
    f_state = state
    results = []
    results_sampled = []
    probs = []
    for i in range(steps):

        y_pred, f_state = qrnn.model(None, state=f_state, return_state=True)

        # Posterior mean.
        y_mean = (
            posterior_mean(y_pred=y_pred, quantile_axis=1, quantiles=quantiles)
            .cpu()
            .numpy()[0]
        )
        results.append(y_mean)

        # P(dbz >= 0)
        prob = (
            probability_larger_than(
                y_pred=y_pred, y=0.0, quantile_axis=1, quantiles=quantiles
            )
            .cpu()
            .numpy()[0]
        )
        probs.append(prob)

        # Sample from posterior
        y_pred_r = sample_posterior(
            y_pred=y_pred, quantile_axis=1, quantiles=quantiles
        )[0, 0]
        results_sampled.append(y_pred_r)
    return np.stack(results), np.stack(probs), np.stack(results_sampled)


MAX_AGE = 16
OVERLAP = 4


def process(model, dataset, output_path):
    """
    Process validation dataset.

    Args:
        model: A trained CIMR model.
        dataset: A test dataset object providing an access to the test
            data.
        output_path: The directory to which to write the results.
    """
    input_iterator = dataset.full_range()

    previous_time = None
    state = None
    age = 0
    input_queue = Queue()

    for model_input, output, y_slice, x_slice, date in input_iterator:

        if previous_time is None:
            time_delta = np.timedelta64(0, "s")
        else:
            time_delta = date - previous_time
        if time_delta > np.timedelta64(20 * 60, "s"):
            state = None
            input_queue = Queue()
        previous_time = date

        if state is None and empty_input(model, model_input):
            continue

        if state is None:
            while input_queue.qsize() > 0:
                _, state = retrieval_step(
                    model, input_queue.get(), y_slice, x_slice, state
                )

        results, state = retrieval_step(model, model_input, y_slice, x_slice, state)

        # Check age of state
        if age == MAX_AGE:
            state = None
        input_queue.put(model_input)
        if input_queue.qsize() > OVERLAP:
            input_queue.get()

        # Add reference values
        y_true = output.detach().cpu().float()
        results["dbz_true"] = (("y", "x"), y_true)
        precip_true = 10 ** ((y_true / 10 - np.log10(200)) / 1.5)
        results["surface_precip_true"] = (("y", "x"), precip_true)

        date = pd.to_datetime(str(date))
        filename = date.strftime("results_%Y_%m_%d_%H%M.nc")
        results.to_netcdf(output_path / filename)


def run(args):
    """
    Extract data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from cimr.data.training_data import CIMRDataset

    #
    # Check and load inputs.
    #

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("Provided model '%s' does not exist.", args.model)
        return 1

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        LOGGER.error("Input '%s' does not exist or is not a directory.", args.input)
        return 1

    output_path = Path(args.output)
    if not output_path.exists() or not output_path.is_dir():
        LOGGER.error("Input '%s' does not exist or is not a directory.", args.output)
        return 1

    qrnn = QRNN.load(model)
    qrnn.model.train(False)

    start_time = args.start_time
    if start_time is not None:
        start_time = np.datetime64(start_time)

    end_time = args.end_time
    if end_time is not None:
        end_time = np.datetime64(end_time)

    test_data = CIMRDataset(input_path, start_time=start_time, end_time=end_time)
    process(qrnn, test_data, output_path)
