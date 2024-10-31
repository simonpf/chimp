"""
chimp.data.extract
==================

Implements the command line interface for extracting CHIMP training data.
"""
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
import logging
from pathlib import Path
from typing import  Any, List, Union

import click

from chimp import areas

LOGGER = logging.getLogger(__name__)


def flatten(lst: Union[List[Any], List[List[Any]]]) -> List[Any]:
    """
    Flatten a potentially nested list.

    Args:
        lst: A list that may contain arbitrarily nested list.

    Return:
        A list containing all elements flattened to a single nesting level.
    """
    res = []
    for elem in lst:
        if isinstance(elem, list):
            res += flatten(elem)
        else:
            res.append(elem)
    return res


@click.argument("input", type=str)
@click.argument("year", type=int)
@click.argument("month", type=str)
@click.argument("days", type=int, nargs=-1)
@click.argument("output", type=str)
@click.option("--time_step", default=15)
@click.option("--domain", default="conus")
@click.option("--conditional", default=None)
@click.option("--path", default=None)
@click.option("--n_processes", default=1)
@click.option("--include_scan_time", default=False)
def cli(
        input,
        year,
        month,
        days,
        output,
        time_step,
        domain,
        conditional,
        path,
        n_processes,
        include_scan_time
):
    """
    Extract training, validation, and test data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from chimp.data.source import get_source
    import chimp.data.gpm
    import chimp.data.cpcir
    import chimp.data.mrms
    import chimp.data.gridsat
    import chimp.data.imerg
    import chimp.data.patmosx
    import chimp.data.ssmi
    import chimp.data.seviri
    import chimp.data.daily_precip
    import chimp.data.baltrad
    import chimp.data.opera
    import chimp.data.s2s
    import chimp.data.wxfm
    import chimp.data.avhrr
    import chimp.data.era5
    import chimp.data.cloudsat

    #
    # Check and load inputs.
    #

    inpt = get_source(input.lower())

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
        domain = areas.get_area(domain)
    except RuntimeError:
        LOGGER.error("Provided domain '%s' is not a known domain.", domain)
        return 1

    time_step = timedelta(minutes=time_step)

    output = Path(output)
    if not output.exists():
        LOGGER.error("Destination must be an existing path!")
        return 1

    if path is not None:
        path = Path(path)
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

    if conditional is not None:
        conditional = Path(conditional)
        if not conditional.exists() or not conditional.is_dir():
            LOGGER.error(
                "If provided, 'conditional' must point to an existing" " directory."
            )
            return 1

    if isinstance(days, int):
        days = [days]

    n_procs = n_processes
    pool = ProcessPoolExecutor(max_workers=n_procs)

    tasks = []
    dates = []
    for month in months:
        if len(days) == 0:
            n_days = monthrange(year, month)[1]
            days_of_month = list(range(1, n_days + 1))
        else:
            days_of_month = days
        for day in days_of_month:
            fargs = [domain, year, month, day, output]
            kwargs = {
                "path": path,
                "time_step": time_step,
                "include_scan_time": include_scan_time,
            }
            if conditional is not None:
                kwargs["conditional"] = conditional
            tasks.append(pool.submit(inpt.process_day, *fargs, **kwargs))
            dates.append((month, day))


    failed = []
    for task, date in zip(tasks, dates):
        res = task.result()
        failed += flatten(res)

    # Write failed days to file
    with open(output / f".{inpt.name}_failed.txt", "w") as output:
        output.write("\n".join([str(path) for path in failed]))
