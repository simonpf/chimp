"""
chimp.data.utils
===============

Utility functions used by the sub-modules of the chimp.data module.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import torch
from pansat import FileRecord
from pansat.time import to_datetime, to_datetime64, to_timedelta64


def scale_slices(
        slices: Union[Tuple[slice, slice], Tuple[int, int, int, int], None],
        rel_scale: float
) -> Tuple[slice, slice]:
    """
    Scale slices to match input scale.

    Args:
        slices: A tuple containing row- and column-slices defining the
            input region to extract with respect to the reference data.
            The tuple can also be of length 4 and contain the start and
            stop values for the rows followed by those for the columns.
            If 'None', slices corresponding to the full row and column
            extent will be returned.
        scale: The scale of the input data relative to the reference data.

    Return:
        A tuple contining the slices scaled to match the relative scale of
        the input data.
    """
    if slices is None:
        return (slice(0, None), slice(0, None))

    if len(slices) == 4:
        slices = (
            slice(slices[0], slices[1]),
            slice(slices[2], slices[3])
        )

    if rel_scale == 1:
        return slices

    row_start = slices[0].start
    row_end = slices[0].stop
    row_step = slices[0].step
    col_start = slices[1].start
    col_end = slices[1].stop
    col_step = slices[1].step
    row_slice = slice(
        int(row_start / rel_scale),
        int(row_end / rel_scale),
        row_step
    )
    col_slice = slice(
        int(col_start / rel_scale),
        int(col_end / rel_scale),
        col_step
    )
    return (row_slice, col_slice)


def round_time(time: np.datetime64, step: np.timedelta64) -> np.datetime64:
    """
    Round time to given time step.

    Args:
        time: A numpy.datetime64 object representing the time to round.
        step: A numpy.timedelta64 object representing the time step to
            which to round the results.
    """
    if isinstance(time, datetime):
        time = to_datetime64(time)
    if isinstance(step, timedelta):
        step = to_timedelta64(step)
    time = time.astype("datetime64[s]")
    step = step.astype("timedelta64[s]")
    rounded = (
        np.datetime64(0, "s")
        + time.astype(np.int64) // step.astype(np.int64) * step
    )
    return rounded


def get_output_filename(
        prefix: str,
        time: np.datetime64,
        time_step: np.timedelta64
):
    """
    Get filename for training sample.

    Args:
        prefix: String specifying the filename prefix.
        time: The observation time.
        minutes: The number of minutes to which to round the time.

    Return:
        A string specifying the filename of the training sample.
    """
    time_r = to_datetime(round_time(time, time_step))
    year = time_r.year
    month = time_r.month
    day = time_r.day
    hour = time_r.hour
    minute = time_r.minute
    filename = f"{prefix}_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    return filename


def records_to_paths(
        path_or_rec: Union[List[FileRecord], List[Path], Path, FileRecord]
) -> Union[Path, List[Path]]:
    """
    Get local path of a file or list of files.

    Args:
        path_or_rec: A path or file record object or list thereof.

    Return:
        A path object pointing to a locally available file or a list thereof.
    """

    if isinstance(path_or_rec, Path):
        return path_or_rec
    if isinstance(path_or_rec, FileRecord):
        return path_or_rec.get().local_path
    if isinstance(path_or_rec, list):
        return [records_to_paths(por) for por in path_or_rec]
    raise RuntimeError(
        "Expected a path, file record or list of those."
    )
