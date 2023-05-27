"""
cimr.data.seviri
================

Functionality for reading and processing SEVIRI data.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside
from pansat.time import to_datetime64
from pansat.download.providers import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances, GOES17L1BRadiances
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr import areas
from cimr.utils import round_time


LOGGER = logging.getLogger(__name__)


def get_output_filename(time):
    """
    Get filename for training sample.

    Args:
        time: The observation time.

    Return:
        A string specifying the filename of the training sample.
    """
    time_15 = round_time(time)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"goes_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    return filename


goes_channels = { # Seviri match
    2:  "geo_01",   # 0.635 um
    3:  "geo_02",   # 0.81 um
    5:  "geo_03",   # 1.64 um
    7:  "geo_04",   # 3.92 um
    8:  "geo_05",   # 6.25 um
    10: "geo_06",   # 7.35 um
    11: "geo_07",   # 8.7 um
    12: "geo_08",   # 9.66 um
    13: "geo_09",   # 10.3 um
    15: "geo_10",   # 12 um
    16: "geo_11"   # 13.4 um
}


def download_and_resample_goes_data(time):
    """
    Download GOES data closest to a given requested time step, resamples it
    and returns the result as an xarray dataset.

    Args:
        time: A datetime object specfiying the time for which to download the
            GOES observations.

    Return:
        An xarray dataset containing the GOES observations resampled to the
        CONUS_4 domain.
    """
    channel_names = [f"C{channel:02}" for channel in goes_channels]
    with TemporaryDirectory() as tmp:

        goes_files = []
        for band in goes_channels:
            provider = GOESAWSProvider(GOES16L1BRadiances("C", band))
            goes_file = provider.get_files_in_range(time, time, start_inclusive=True)[-1]
            local_file = Path(tmp) / goes_file
            provider.download_file(goes_file, local_file)
            goes_files.append(local_file)

        scene = Scene([str(filename) for filename in goes_files], reader="abi_l1b")
        scene.load(channel_names)
        scene = scene.resample(areas.CONUS_4)
        data = scene.to_xarray_dataset().compute()

        goes_files = []
        for band in goes_channels:
            provider = GOESAWSProvider(GOES17L1BRadiances("C", band))
            goes_file = provider.get_files_in_range(time, time, start_inclusive=True)[-1]
            local_file = Path(tmp) / goes_file
            provider.download_file(goes_file, local_file)
            goes_files.append(local_file)

        if len(goes_files) > 0:

            scene = Scene([str(filename) for filename in goes_files], reader="abi_l1b")
            scene.load(channel_names)
            scene = scene.resample(areas.CONUS_4)
            data_g17 = scene.to_xarray_dataset().compute()

            for channel in channel_names:
                data[channel].data[:, :500] = data_g17[channel].data[:, :500]

        rename_dict = dict(zip(channel_names, goes_channels.values()))

        data = data.drop_vars(["crs"])
        start_time = data.attrs["start_time"]
        d_t = data.attrs["end_time"] - data.attrs["start_time"]
        data.attrs = {}
        data.attrs["time"] = to_datetime64(start_time + 0.5 * d_t)

        for var in data:
            data[var].attrs = {}
        return data.rename(rename_dict)


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            GOES observations.
        output_folder: The folder to which to write the training data.

    """
    dataset = dataset.copy()
    filename = get_output_filename(dataset.attrs["time"].item())
    output_filename = Path(output_folder) / filename
    dataset.attrs["time"] = str(dataset.attrs["time"])

    encoding = {}

    for channel in ["geo_01", "geo_02", "geo_03"]:
        dataset[channel].data[:] = np.minimum(dataset[channel].data, 127)
        encoding[channel] = {
            "dtype": "uint8",
            "_FillValue": 255,
            "scale_factor": 0.5,
            "zlib": True
        }

    for channel in range(4, 12):
        channel = f"geo_{channel:02}"
        dataset[channel].data[:] = np.clip(dataset[channel].data, 195, 323)
        encoding[channel] = {
            "dtype": "uint8",
            "scale_factor": 0.5,
            "add_offset": 195,
            "_FillValue": 255,
            "zlib": True
        }
    dataset.to_netcdf(output_filename, encoding=encoding)


def process_day(
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(minutes=15)
):
    """
    Extract training data from a day of SEVIRI observations.

    Args:
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
    """
    output_folder = Path(output_folder) / "geo"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    existing_files = [
        f.name for f in output_folder.glob(f"goes_{year}{month:02}{day:02}*.nc")
    ]

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    time = start_time
    while time < end_time:

        output_filename = get_output_filename(time)
        print(output_filename, (output_folder / output_filename).exists())
        if not (output_folder / output_filename).exists():
            dataset = download_and_resample_goes_data(time)
            save_file(dataset, output_folder)

        time = time + time_step
