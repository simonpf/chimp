"""
chimp.utils
===========

Utility module containing functions shared across the package.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from pansat.time import to_datetime


def get_available_times(path):
    """
    Get times of available CHIMP files within a given directory.

    Args:
        path: A string or path object identifying a folder containing
            CHIMP files.

    Return:
        A set containing the times corresponding to the CHIMP files
        in the given folder.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            "'path' must to an existing directory."
        )
    available_files = path.glob("*????????_??_??.nc")
    available_times = [
        datetime.strptime(filename.name[-17:-3], "%Y%m%d_%H_%M")
        for filename in available_files
    ]
    return available_times


def get_date(path: Union[Path, str]) -> np.datetime64:
    """
    Extract date from a training data filename.

    Args:
        path: A path object or string identifying a CHIMP data file.

    Return:
        Numpy datetime64 object containing the time corresponding to
        the training sample.
    """
    if isinstance(path, str):
        path = Path(path)

    *_, yearmonthday, hour, minute = path.stem.split("_")
    year = yearmonthday[:4]
    month = yearmonthday[4:6]
    day = yearmonthday[6:]
    return np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")
