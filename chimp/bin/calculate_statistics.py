"""
chimp.bin.calculate_statistics
=============================+

This sub-module implements a command-line application to calculate statistics
of training data.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import List

import numpy as np
from rich.progress import track
import xarray as xr


import chimp
import chimp.data
from chimp.logging import CONSOLE


LOGGER = logging.getLogger(__file__)


class InputDataStatistics:
    """
    Computation of mean, min and max of training input data.
    """
    def __init__(self, inpt):
        """
        Args:
            inpt: An object representing the training input for which the
                the statistics are computed.
        """
        self.inpt = inpt
        self.sums = None
        self.counts = None
        self.maxs = None
        self.mins = None

    def calculate(self, input_filename: Path) -> None:
        """
        Calculate statistics for a single file.

        Args:
            input_filename: Path pointing to a training data file for which
                to calculate the statistics.
        """
        with xr.open_dataset(input_filename) as data:
            var = self.inpt.variables
            inpt_data = data[var].data

            sums = []
            counts = []
            mins = []
            maxs = []

            for ch_ind in range(inpt_data.shape[0]):
                ch_data = inpt_data[ch_ind]
                valid = np.isfinite(ch_data)

                if np.all(~valid):
                    continue

                ch_data = ch_data[valid]
                sums.append(ch_data.sum())
                counts.append(valid.sum())
                mins.append(ch_data.min())
                maxs.append(ch_data.max())

            sums = np.stack(sums)
            counts = np.stack(counts)
            maxs = np.stack(maxs)
            mins = np.stack(mins)

        if self.sums is None:
            self.sums = sums
            self.counts = counts
            self.maxs = maxs
            self.mins = mins
        else:
            self.sums += sums
            self.counts += counts
            self.maxs = np.maximum(self.maxs, maxs)
            self.mins = np.minimum(self.mins, mins)

    def merge(self, other: "InputDataStatistics") -> None:
        """
        Merge statistics with those calculate from another input data statistics
        object.

        Args:
            other: The other statistics object.
        """
        if self.sums is None:
            self.sums = other.sums
            self.counts = other.counts
            self.maxs = other.maxs
            self.mins = other.mins
        elif other.sums is not None:
            self.sums += other.sums
            self.counts += other.counts
            self.maxs = np.maximum(self.maxs, other.maxs)
            self.mins = np.minimum(self.mins, other.mins)

    def to_netcdf(self, path: Path) -> None:
        """
        Write calculated statistics to a NetCDF file.

        Args:
            path: Path pointing to a directory or file to which to write
                the statistics.
        """
        if self.mins is None:
            LOGGER.warning(
                "Your are trying to write empty input data statistics."
                " No file produced."
            )

        path = Path(path)
        if path.is_dir():
            path = path / "input_statistics.nc"

        dataset = xr.Dataset({
            "tb_min": (("channels",), self.mins),
            "tb_max": (("channels",), self.maxs),
            "tb_mean": (("channels",), self.sums / self.counts)
        })
        dataset.to_netcdf(path)


def process_file(inpt: chimp.data.Input, input_file: Path) -> InputDataStatistics:
    """
    Calculate input data statistics for a single file.

    Args:
        inpt: The input object representing the input data for which statistics
            are being calculated.
        input_file: Path to the file to process.

    Return:
        An InputDataStatistics object containing the  statistics from the input
        data file.
    """
    stats = InputDataStatistics(inpt)
    stats.calculate(input_file)
    return stats


def process_files(
        inpt : chimp.data.Input,
        files : List[Path],
        n_processes : int
) -> InputDataStatistics:
    """
    Process multiple input files in parallel.
    """
    pool = ProcessPoolExecutor(max_workers=n_processes)

    tasks = []
    for path in files:
        task = pool.submit(process_file, inpt, path)
        task._inpt_file = path
        tasks.append(task)

    stats = None
    for task in track(
            as_completed(tasks),
            description="Calculating statistics: ",
            total=len(tasks),
            console=CONSOLE
    ):
        try:
            res = task.result()

            if stats is None:
                stats = res
            else:
                stats.merge(res)
        except Exception:
            LOGGER.exception(
                "An error occurred during the processing of file "
                f" {task._inpt_file}."
            )
    return stats


def add_parser(subparsers):
    """
    Add parser for 'calculate_statistics' command to top-level parser. This
    function is called from the top-level parser defined in 'chimp.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "stats",
        help="Calculate training data statistics.",
        description=(
            """
            Calculate summary statistics of the training data.
            """
        ),
    )
    parser.add_argument(
        "source",
        metavar="input/reference",
        type=str,
        help=(
            "The name of the input data or reference data source."
        )
    )
    parser.add_argument(
        "path",
        metavar="path",
        type=str,
        help="The folder containing the training files.",
    )
    parser.add_argument(
        "--n_processes",
        metavar="N",
        type=int,
        help=(
            "The number of processes to use for the calcuations of the"
            " statistics."
        ),
        default=4
    )
    parser.set_defaults(func=run)


def run(args):
    """
    Runs the calculation of the statistics.
    """
    from chimp.data import inputs

    inpt = getattr(inputs, args.source.upper(), None)
    if inpt is None:
        LOGGER.error(
            f"Provided input '{args.source}' is not a know input or reference "
            " data source."
        )
        return 1

    path = Path(args.path)
    if not path.exists():
        LOGGER.error(
            f"Provided path '{path}' does not exist."
        )

    files = sorted(list(path.glob("*.nc")))
    stats = process_files(inpt, files, args.n_processes)

    if stats is None:
        LOGGER.error(
            f"No files found in the provided data folder {path}."
        )
        return 1

    stats.to_netcdf(".")
