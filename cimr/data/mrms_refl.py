"""
cimr.data.mrms_refl
===================

Functionality to download and extract MRMS reflectivity reference data.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from h5py import File
import numpy as np
import pandas as pd
import dask.array as da
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
from scipy.stats import binned_statistic_2d
from pansat.download.providers import IowaStateProvider
from pansat.products.ground_based.mrms import mrms_reflectivity
from pansat.time import to_datetime64
import xarray as xr

from cimr.utils import round_time, get_available_times
from cimr import areas


def resample_mrms_data(dataset):
    """
    Resample MRMS data to CIMR grid.

    Args:
        dataset: 'xarray.Dataset' containing the MRMS surface precip and
            radar quality index.
    """
    lons_out, lats_out = areas.CONUS_4.get_lonlats()

    lons_out = lons_out[0]
    lon_bins = np.zeros(lons_out.size + 1)
    lon_bins[1:-1] = 0.5 * (lons_out[1:] + lons_out[:-1])
    lon_bins[0] = lon_bins[1] - (lon_bins[2] - lon_bins[1])
    lon_bins[-1] = lon_bins[-2] + lon_bins[-2] - lon_bins[-3]

    lats_out = lats_out[..., 0][::-1]
    lat_bins = np.zeros(lats_out.size + 1)
    lat_bins[1:-1] = 0.5 * (lats_out[1:] + lats_out[:-1])
    lat_bins[0] = lat_bins[1] - (lat_bins[2] - lat_bins[1])
    lat_bins[-1] = lat_bins[-2] + lat_bins[-2] - lat_bins[-3]

    lons, lats = areas.MRMS.get_lonlats()
    refl = dataset.reflectivity.data
    valid = refl >= -900
    refl = binned_statistic_2d(
        lats[valid],
        lons[valid],
        refl[valid],
        bins=(lat_bins, lon_bins)
    )[0][::-1]

    dataset_r = xr.Dataset({
        "latitude": (("latitude"), lats_out),
        "longitude": (("longitude"), lons_out),
        "reflectivity": (("latitude", "longitude"), refl),
        "time": ((), dataset.time.data)
    })
    return dataset_r


def get_output_filename(time):
    """
    Get filename of training data file for given time.

    Args:
        time: The time of the reference data

    Return:
        A string containing the filename of the output file.

    """
    time_15 = round_time(time, minutes=15)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute
    filename = f"radar_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    return filename


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the MRMS observations.
        output_folder: The folder to which to write the training data.

    """

    filename = get_output_filename(dataset.time.data.item())
    output_filename = Path(output_folder) / filename

    comp = {"zlib": True}
    encoding = {var: comp for var in dataset.variables.keys()}

    refl = dataset.reflectivity.data
    dataset.reflectivity.data = np.maximum(refl, -20)
    encoding["reflectivity"] = {
        "scale_factor": 1 / 200,
        "add_offset": -20,
        "dtype": "uint16",
        "zlib": True,
        "_FillValue": -1
    }

    dataset.to_netcdf(output_filename, encoding=encoding)


def process_day(
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(minutes=15),
        conditional=None):
    """
    Extract training data from a day of MRMS measurements.

    Args:
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
        time_step: Time step defining the temporal resolution at which to extract
            training samples.
        conditional: A folder containing CIMR files from a different source.
            CIMR will only extract reference data that matches the time stamps
            for which files from the other source are available.
    """
    output_folder = Path(output_folder) / "mrms_refl"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    reflectivity_provider = IowaStateProvider(mrms_reflectivity)

    if conditional is not None:
        available_times = set(get_available_times(conditional))
    else:
        available_times = None

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    time = start_time
    files = []
    while time < end_time:

        time_r = round_time(time, minutes=time_step.seconds // 60)
        if available_times is None or time_r in available_times:
            output_filename = get_output_filename(to_datetime64(time))
            if not (output_folder / output_filename).exists():
                files += reflectivity_provider.get_files_in_range(
                    time,
                    time,
                    start_inclusive=True
                )[-1:]

        time = time + time_step

    print(files)

    for reflectivity_file in files:
        with TemporaryDirectory() as tmp:
            reflectivity_file = Path(tmp) / reflectivity_file
            reflectivity_provider.download_file(
                reflectivity_file.name,
                reflectivity_file
            )
            reflectivity_data = mrms_reflectivity.open(reflectivity_file)
            reflectivity_data = resample_mrms_data(reflectivity_data)
            save_file(reflectivity_data, output_folder)
