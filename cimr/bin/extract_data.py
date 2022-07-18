"""
=====================
cimr.bin.extract_data
=====================

This sub-module implements the cimr CLI to extract training data.
"""
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'cimr.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "extract_data",
        help="Extraction of training data.",
        description=(
            """
            Extract training data for cimr.
            """
        ),
    )
    parser.add_argument(
        "sensor",
        metavar="sensor",
        type=str,
        help="The name of the sensor from which to extract the training data.",
    )
    parser.add_argument(
        "year",
        metavar="year",
        type=int,
        help="The year for which to extract the data.",
    )
    parser.add_argument(
        "month",
        metavar="year",
        type=int,
        help="The month for which to extract the data.",
    )
    parser.add_argument(
        "day",
        metavar="year",
        type=int,
        nargs="*",
        help=("The days for which to extract the data. If omitted data for all"
              "days of the month will be extracted.")
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Destination to store the extracted data. "
    )
    parser.add_argument(
        "--n_processes",
        metavar="n",
        type=int,
        default=1,
        help="The number of processes to use for the data extraction.",
    )
    parser.set_defaults(func=run)


def process_file(
    input_file,
    output_file,
    model,
    targets,
    gradients,
    device,
    log_queue,
    preserve_structure=False,
    fmt=None,
    sensor=None,
):
    """
    Process input file.

    Args:
        input_file: Path pointing to the input file.
        output_file: Path to the file to which to store the results.
        model: The GPROF-NN model with which to run the retrieval.
        targets: List of the targets to retrieve.
        gradients: Whether or not to do a special run to calculate
            gradients of the retrieval.
        device: The device on which to run the retrieval
        log_queue: Queue object to use for multi process logging.
    """
    gprof_nn.logging.configure_queue_logging(log_queue)
    logger = logging.getLogger(__name__)
    logger.info("Processing file %s.", input_file)
    xrnn = QRNN.load(model)
    if targets is not None:
        xrnn.set_targets(targets)
    driver = RetrievalDriver
    if gradients:
        driver = RetrievalGradientDriver
    retrieval = driver(
        input_file,
        xrnn,
        output_file=output_file,
        device=device,
        preserve_structure=preserve_structure,
        sensor=sensor,
        output_format=fmt,
        tile=False,
    )
    retrieval.run()


def run(args):
    """
    Extract data.

    Args:
        args: The namespace object provided by the top-level parser.
    """

    #
    # Check and load inputs.
    #

    module_name = f"cimr.data.{args.sensor.lower()}"
    module = importlib.import_module(module_name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        LOGGER.error(
            "Sensor '%s' currently not supported."
        )
        return 1

    year = args.year
    month = args.month
    days = args.day
    if len(days) == 0:
        days = list(range(1, 32))

    output = Path(args.output)
    if not output.exists():
        LOGGER.error(
            "Destination must be an existing path!"
        )
        return 1


    n_procs = args.n_processes
    pool = ThreadPoolExecutor(max_workers=n_procs)

    tasks = []
    for day in days:
        tasks.append(pool.submit(module.process_day, year, month, day, output))

    for task, day in zip(tasks, days):
        task.result()



