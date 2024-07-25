"""
==================
chimp.bin.forecast
==================

This sub-module implements the chimp CLI to run forecasts.
"""
import logging
from pathlib import Path
from typing import List, Union


LOGGER = logging.getLogger(__file__)

def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'chimp.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "forecast",
        help="Produce a precipitation forecast.",
        description=(
            """
            Produce a precipitation forecast.
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
        nargs="+",
        help="Path to the test data or a set of files.",
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Destination to store the test results.",
    )
    parser.add_argument(
        "start_time",
        metavar="start",
        type=str,
        help="First time for which to perform a forecast.",
    )
    parser.add_argument(
        "end_time",
        metavar="end",
        type=str,
        help="Last time for which to perform a forecast.",
    )
    parser.add_argument(
        "--step",
        metavar="min",
        type=int,
        help="The time step between consecutive forecasts.",
        default=15,
    )
    parser.set_defaults(func=run)


def process_time(qrnn, input_data, time, output):
    """
    Makes a forecast for a given initialization time.

    Args:
        time: The time at which to initialize the forecast.
        output: The folder to which to write the results.
    """
    from pansat.time import to_datetime
    from chimp.processing import make_forecast

    time_py = to_datetime(time)
    filename = time_py.strftime("results_%Y_%m_%d_%H%M.nc")
    result = make_forecast(qrnn, input_data, time, 16, 16)
    if result is not None:
        result.to_netcdf(output / filename)


def run(args):
    """
    Run the forecasts.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    import numpy as np
    from quantnn.qrnn import QRNN
    from chimp.data.training_data import SingleStepDataset

    #
    # Check and load inputs.
    #

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("Provided model '%s' does not exist.", args.model)
        return 1

    inputs = [Path(input) for input in args.input]
    if len(inputs) == 1:
        if inputs[0].exists() and inputs[0].is_file():
            input_path: Union[Path, list[Path]] = inputs
        elif inputs[0].exists() and inputs[0].is_dir():
            input_path = inputs[0]
        else:
            LOGGER.error(
                "Input '%s' does not exist or is not a file or directory.",
                args.input,
            )
            return 1
    else:
        for input in inputs:
            if not input.exists() or not input.is_file():
                LOGGER.error("Input '%s' does not exist or is not a file.", input)
                return 1
        input_path = inputs

    output_path = Path(args.output)
    if not output_path.exists() or not output_path.is_dir():
        LOGGER.error("Input '%s' does not exist or is not a directory.", args.output)
        return 1

    start_time = np.datetime64(args.start_time)
    end_time = np.datetime64(args.end_time)
    step = np.timedelta64(args.step * 60, "s")
    qrnn = QRNN.load(model)
    input_data = SingleStepDataset(input_path, sources=qrnn.model.sources)
    for time in np.arange(start_time, end_time, step):
        process_time(qrnn, input_data, time, output_path)

    return 0
