"""
=================
cimr.bin.forecast
=================

This sub-module implements the cimr CLI to run forecasts.
"""
import logging
from pathlib import Path


LOGGER = logging.getLogger(__file__)

def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'cimr.bin'.

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
        help="Path to the test data.",
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
    from cimr.processing import make_forecast

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

    start_time = np.datetime64(args.start_time)
    end_time = np.datetime64(args.end_time)
    step = np.timedelta64(args.step * 60, "s")
    qrnn = QRNN.load(model)
    input_data = CIMRDataset(input_path, sources=qrnn.model.sources)
    for time in np.arange(start_time, end_time, step):
        process_time(qrnn, input_data, time, output_path)

    return 0
