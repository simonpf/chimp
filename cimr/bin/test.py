"""
=============
cimr.bin.test
=============

This sub-module implements the cimr CLI to run CIMR models on
test data.
"""
import logging
from pathlib import Path
from queue import Queue

import numpy as np
import pandas as pd
from torch import nn
from quantnn import QRNN
from quantnn.packed_tensor import PackedTensor
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
        "--inputs",
        metavar="name",
        type=str,
        nargs="+",
        help="Names of the inputs sources to use.",
    )
    parser.add_argument(
        "--reference_data",
        metavar="name",
        type=str,
        help="Name of the reference data.",
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
    input_iterator = dataset.full_domain()

    previous_time = None
    state = None
    age = 0
    input_queue = Queue()

    for time, x, y in input_iterator:

        results, _ = retrieval_step(model, (x, y), state)

        results["surface_precip_ref"] = (("y", "x"), y["surface_precip"])

        date = pd.to_datetime(str(time))
        filename = date.strftime("results_%Y_%m_%d_%H%M.nc")
        results.to_netcdf(output_path / filename)


def run(args):
    """
    Run the retrieval.

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

    test_data = CIMRDataset(
        input_path,
        inputs=args.inputs,
        reference_data=args.reference_data,
        start_time=start_time,
        end_time=end_time
    )
    process(qrnn, test_data, output_path)

    return 0
