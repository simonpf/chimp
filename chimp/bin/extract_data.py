"""
======================
chimp.bin.extract_data
======================

This sub-module implements the chimp CLI to extract training data.
"""
from calendar import monthrange
from datetime import timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from chimp import areas

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_data' command to top-level parser. This function
    is called from the top-level parser defined in 'chimp.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "extract_data",
        help="Extraction of training data.",
        description=(
            """
            Extract training data for chimp.
            """
        ),
    )
    parser.add_argument(
        "input",
        metavar="input",
        type=str,
        help="The name of the input data to extract.",
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
        type=str,
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
            "Path containing CHIMP files from a different source. If provided "
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
    parser.add_argument(
        "--include_scan_time", action="store_true",
        help="Include resampled scan time in retrieval inputs."

    )
    parser.set_defaults(func=run)


def run(args):
    """
    Extract data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from chimp.data.source import get_source
    import chimp.data.gpm
    import chimp.data.cpcir
    import chimp.data.mrms
    import chimp.data.gridsat
    import chimp.data.patmosx
    import chimp.data.ssmi
    import chimp.data.seviri
    import chimp.data.daily_precip
    import chimp.data.baltrad

    #
    # Check and load inputs.
    #

    inpt = get_source(args.input.lower())

    year = args.year
    month = args.month
    if month in ["*", "?"]:
        months = list(range(1, 13))
    else:
        try:
            months = [int(month)]
        except Exception():
            LOGGER.error(
                "Month argument must be an integer identifying the month of "
                " the year or '*'."
            )
            return 1


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
    include_scan_time = args.include_scan_time
    pool = ThreadPoolExecutor(max_workers=n_procs)

    tasks = []
    dates = []
    for month in months:
        days = args.day
        if len(days) == 0:
            n_days = monthrange(year, month)[1]
            days = list(range(1, n_days + 1))
        for day in days:
            fargs = [domain, year, month, day, output]
            kwargs = {
                "path": path,
                "time_step": time_step,
                "include_scan_time": include_scan_time
            }
            if conditional is not None:
                kwargs["conditional"] = conditional
            tasks.append(pool.submit(inpt.process_day, *fargs, **kwargs))
            dates.append((month, day))

    failed_days = []

    for task, date in zip(tasks, dates):
        try:
            ret = task.result()
        except Exception as e:
            LOGGER.exception(
                "The following error was encountered while processing file '%s': %s %s",
                f"{year}-{month:02}-{day:02}",
                type(e),
                e)
            ret = 1

        if ret is not None and ret > 0:
            month, day = date
            failed_days.append((year, month, day))


    # Write failed days to file
    with open(output / f".{inpt.name}_failed.txt", "w") as failed:
        for year, month, day in failed_days:
            failed.write(f"{year} {month} {day}\n")
