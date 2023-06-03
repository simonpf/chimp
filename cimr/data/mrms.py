"""
cimr.data.mrms
==============

Functionality to download and extract MRMS reference data.
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
from pansat.products.ground_based.mrms import mrms_precip_rate, mrms_radar_quality_index
from pansat.time import to_datetime64
import xarray as xr

from cimr.utils import round_time
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
    sfp = dataset.surface_precip.data
    valid = sfp >= 0.0
    surface_precip = binned_statistic_2d(
        lats[valid],
        lons[valid],
        sfp[valid],
        bins=(lat_bins, lon_bins)
    )[0][::-1]
    rqi = dataset.rqi.data
    valid = rqi >= 0.0
    rqi = binned_statistic_2d(
        lats[valid],
        lons[valid],
        rqi[valid],
        bins=(lat_bins, lon_bins)
    )[0][::-1]

    dataset_r = xr.Dataset({
        "latitude": (("latitude"), lats_out),
        "longitude": (("longitude"), lons_out),
        "surface_precip": (("latitude", "longitude"), surface_precip),
        "rqi": (("latitude", "longitude"), rqi),
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
    time_15 = round_time(time)
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

    surface_precip = np.minimum(dataset.surface_precip.data, 300)
    dataset.surface_precip.data = surface_precip
    encoding["surface_precip"] = {
        "scale_factor": 1 / 100,
        "dtype": "int16",
        "zlib": True,
        "_FillValue": -1
    }

    dataset.surface_precip.data = surface_precip
    encoding["rqi"] = {
        "scale_factor": 1 / 128,
        "dtype": "int8",
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
        time_step=timedelta(minutes=15)):
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
    """
    output_folder = Path(output_folder) / "mrms"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)


    precip_rate_provider = IowaStateProvider(mrms_precip_rate)
    rqi_provider = IowaStateProvider(mrms_radar_quality_index)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    time = start_time
    files = []
    while time < end_time:
        output_filename = get_output_filename(to_datetime64(time))
        if not (output_folder / output_filename).exists():
            files += zip(
                precip_rate_provider.get_files_in_range(time, time, start_inclusive=True)[-1:],
                rqi_provider.get_files_in_range(time, time, start_inclusive=True)[-1:]
            )
        print(files, time)
        time = time + time_step

    for precip_rate_file, rqi_file in files:
        with TemporaryDirectory() as tmp:
            precip_rate_file = Path(tmp) / precip_rate_file
            precip_rate_provider.download_file(precip_rate_file.name, precip_rate_file)

            rqi_file = Path(tmp) / rqi_file
            rqi_provider.download_file(rqi_file.name, rqi_file)

            precip_rate_data = mrms_precip_rate.open(precip_rate_file)
            rqi_data = mrms_radar_quality_index.open(rqi_file)

            dataset = xr.merge([precip_rate_data, rqi_data]).rename({
                "precip_rate": "surface_precip",
                "radar_quality_index": "rqi"
            })
            dataset = resample_mrms_data(dataset)
            save_file(dataset, output_folder)
