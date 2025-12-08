"""
Functionality to calculate training data statistics.
"""
import logging
from pathlib import Path
from typing import List

import click
import numpy as np
from tqdm import tqdm
import xarray as xr

from chimp.data import (
    get_input_dataset,
    get_reference_dataset,
)
from chimp.utils import get_date


LOGGER = logging.getLogger(__name__)


@click.argument("path", type=str)
@click.argument("reference_datasets", type=str, nargs=-1)
@click.argument("output_path", type=str)
def calculate_spatial_statistics(
        path: Path,
        reference_datasets: List[str],
        output_path: Path
):
    """
    Calculate training statistics.

    Args:
        path: Path object pointing to the root directory of the training data.
        input_datasets: Lists of the name of the input datasets.
        reference_datasets: Lists of the name of the reference datasets.
    """
    path = Path(path)
    output_path = Path(output_path)

    #input_datasets = np.array(
    #    [get_input_dataset(input_dataset) for input_dataset in input_datasets]
    #)
    reference_datasets = np.array(
        [
            get_reference_dataset(reference_dataset)
            for reference_dataset in reference_datasets
        ]
    )

    for rds in reference_datasets:
        files = rds.find_training_files(path)[1]
        if len(files):
            LOGGER.warning(
                "Found no files for reference dataset %s.",
                rds.name
            )
        else:
            LOGGER.info(
                "Found %s files for reference dataset %s.",
                len(files), rds.name
            )

        times = np.array([get_date(path) for path in files])
        min_time = times.min().astype("datetime64[M]")
        max_time = times.max().astype("datetime64[M]")
        time_bins = np.arange(min_time, max_time + np.timedelta64(1, "M"))

        sums = None
        cts = None
        for path in tqdm(files):
            date = get_date(path)
            try:
                with xr.open_dataset(path) as data:
                    trg = data[rds.targets[0].name].data
                    valid = np.isfinite(trg)


                    time_bin = (date - min_time).astype("timedelta64[M]").astype("uint64")

                    if sums is None:
                        sums = np.zeros((time_bins.size - 1,) + trg.shape, dtype="float32")
                        cts = np.zeros((time_bins.size - 1,) + trg.shape, dtype="float32")

                    sums[time_bin] += np.nan_to_num(trg, nan=0.0, copy=True)
                    cts[time_bin] += valid.astype(np.float32)

            except Exception:
                LOGGER.exception("Encountered an error when opening file %s.", str(path))

        monthly_means = sums / cts
        mean = sums.sum(0) / cts.sum(0)

        time = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])

        results = xr.Dataset({
            "time": (("time"), time),
            "monthly_means": (("time", "y", "x"), monthly_means),
            "mean": (("y", "x"), mean),
        })
        results.to_netcdf(output_path / f"{rds.name}.nc")
