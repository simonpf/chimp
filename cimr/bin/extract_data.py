"""
=====================
cimr.bin.extract_data
=====================

This sub-module implements the cimr CLI to extract training data.
"""
from calendar import monthrange
from datetime import timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path

from cimr import areas

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
        metavar="month",
        type=int,
        help="The month for which to extract the data.",
    )
    parser.add_argument(
        "day",
        metavar="day",
        type=int,
        nargs="*",
        help=(
            "The days for which to extract the data. If omitted data for all"
            "days of the month will be extracted."
        ),
    )
    parser.add_argument(
        "output",
        metavar="output",
        type=str,
        help="Destination to store the extracted data. ",
    )
    parser.add_argument(
        "--time_step",
        metavar="minutes",
        type=int,
        default=15,
        help="The time difference between consecutive retrieval steps.",
    )
    parser.add_argument(
        "--domain",
        metavar="name",
        type=str,
        default="CONUS",
        help="The name of the domain for which to extract data.",
    )
    parser.add_argument(
        "--conditional",
        metavar="name",
        type=str,
        default=None,
        help=(
            "Path containing CIMR files from a different source. If provided "
            " only files will be extracted for which a matching file from the "
            " other data already exists."
        )
    )
    parser.add_argument(
        "--path", metavar="path", type=str, help="Location of local input data."
    )
    parser.add_argument(
        "--n_processes",
        metavar="n",
        type=int,
        default=1,
        help="The number of processes to use for the data extraction.",
    )
    parser.set_defaults(func=run)


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
        LOGGER.error("Sensor '%s' currently not supported.")
        return 1

    year = args.year
    month = args.month
    days = args.day
    if len(days) == 0:
        n_days = monthrange(year, month)[1]
        days = list(range(1, n_days + 1))

    try:
        domain = getattr(areas, args.domain.upper())
    except AttributeError:
        LOGGER.error(
            "Provided domain '%s' is not a known domain.",
            domain
        )
        return 1

    time_step = timedelta(minutes=args.time_step)


    output = Path(args.output)
    if not output.exists():
        LOGGER.error("Destination must be an existing path!")
        return 1

    if args.path is not None:
        path = Path(args.path)
        if not path.exists():
            LOGGER.error(
                """
                If provided, 'path' must point to an existing directory. Currently
                it points to %s.
                """,
                path,
            )
            return 1
    else:
        path = None

    conditional = args.conditional
    if conditional is not None:
        conditional = Path(conditional)
        if not conditional.exists() or not conditional.is_dir():
            LOGGER.error(
                "If provided, 'conditional' must point to an existing"
                " directory."
            )
            return 1

    n_procs = args.n_processes
    pool = ThreadPoolExecutor(max_workers=n_procs)

    tasks = []
    for day in days:
        args = [domain, year, month, day, output]
        kwargs = {"path": path, "time_step": time_step}
        if conditional is not None:
            kwargs["conditional"] = conditional
        tasks.append(pool.submit(module.process_day, *args, **kwargs))

    for task, day in zip(tasks, days):
        try:
            task.result()
        except Exception as e:
            LOGGER.exception(
                "The following error was encountered while processing file '%s': %s %s",
                f"{year}-{month:02}-{day:02}",
                type(e),
                e)
