"""
cimr.utils
==========

Utility module containing functions shared across the package.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def round_time(time, minutes=15):
    """
    Round time to closest 15 minutes.

    Args:
        time: A representation of a time that can be parsed by
            'pandas.Timestamp'

    Return:
        A 'datetime.datetime' object representing the rounded time.
    """

    time = pd.Timestamp(time).to_pydatetime()
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute

    minute_r = minutes * np.round(minute / minutes)
    time_r = datetime(year, month, day, hour) + timedelta(minutes=minute_r)

    return time_r


def extract_tbs_mw(path):
    """
    Extract all valis TBs from microwave data.

    Args:
        path: The path containing the microwave training data.
    """
    mw_90_vals = []
    mw_160_vals = []
    mw_183_vals = []
    files = Path(path).glob("*.nc")
    for filename in files:
        data = xr.load_dataset(filename)
        if "mw_90" in data:
            tbs = data["mw_90"].data
            valid = np.all(np.isfinite(tbs), axis=-1)
            mw_90_vals.append(tbs[valid])
        if "mw_160" in data:
            tbs = data["mw_160"].data
            valid = np.all(np.isfinite(tbs), axis=-1)
            mw_160_vals.append(tbs[valid])
        if "mw_183" in data:
            tbs = data["mw_183"].data
            valid = np.all(np.isfinite(tbs), axis=-1)
            mw_183_vals.append(tbs[valid])

    mw_90_vals = np.concatenate(mw_90_vals, axis=0)
    mw_160_vals = np.concatenate(mw_160_vals, axis=0)
    mw_183_vals = np.concatenate(mw_183_vals, axis=0)
    return mw_90_vals, mw_160_vals, mw_183_vals


def extract_obs_visir(path):
    """
    Extract all valid observations values from VIS/IR data.

    Args:
        path: The path containing the microwave training data.
    """
    vals = {}

    files = Path(path).glob("*.nc")
    for filename in files:
        data = xr.load_dataset(filename)
        for i in range(5):
            name = f"visir_{(i + 1):02}"
            obs = data[name].data
            valid = np.isfinite(obs)
            vals.setdefault(name, []).append(obs[valid])

    for i in range(5):
        name = f"visir_{(i + 1):02}"
        vals[name] = np.concatenate(vals[name], axis=0)

    return vals


def extract_obs_geo(path, samples_per_file=100):
    """
    Extract all valid observations values from GEO data.

    Args:
        path: The path containing the training data.
    """
    vals = {}

    files = Path(path).glob("*.nc")
    for filename in files:
        data = xr.load_dataset(filename)
        for i in range(11):
            name = f"geo_{(i + 1):02}"
            obs = data[name].data
            valid = np.isfinite(obs)
            samples = np.random.choice(obs[valid], size=samples_per_file)
            vals.setdefault(name, []).append(obs[valid])

    for i in range(11):
        name = f"geo_{(i + 1):02}"
        vals[name] = np.concatenate(vals[name], axis=0)

    return vals


def get_available_times(path):
    """
    Get times of available CIMR files within a given directory.

    Args:
        path: A string or path object identifying a folder containing
            CIMR files.

    Return:
        A set containing the times corresponding to the CIMR files
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
