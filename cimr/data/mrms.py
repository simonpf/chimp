"""
cimr.data.mrms
==============

This module implements the functionality to download and resample
MRMS surface precipitation data to be used as reference data for
training CIMR retrievals.
"""
from datetime import datetime, timedelta
import logging
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
from pansat.products.ground_based import mrms
from pansat.time import TimeRange

from pansat.time import to_datetime64
import xarray as xr

from cimr.utils import round_time
from cimr import areas


LOGGER = logging.getLogger(__name__)

PRECIP_TYPES = {
    "No rain": [0.0],
    "Stratiform": [1.0, 2.0, 10.0, 91.0],
    "Convective": [6.0, 96.0],
    "Hail": [7.0],
    "Snow": [3.0, 4.0],
}


def resample_mrms_data(dataset: xr.Dataset) -> xr.Dataset:
    """
    Resample MRMS data to CIMR grid.

    Args:
        dataset: 'xarray.Dataset' containing the MRMS surface precip and
            radar quality index.

    Return:
        A new dataset containing the surface precip and RQI data
        resampled to the CONUS 4-km domain.
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

    precip_flag = dataset.precip_type.data
    precip_type_cts = []
    names = []
    for name, flags in PRECIP_TYPES.items():
        type_field = None
        for type_flag in flags:
            if type_field is None:
                type_field = np.isclose(precip_flag, type_flag)
            else:
                type_field += np.isclose(precip_flag, type_flag)
        cts = binned_statistic_2d(
            lats[valid],
            lons[valid],
            type_field[valid],
            bins=(lat_bins, lon_bins)
        )[0][::-1]
        precip_type_cts.append(cts)
    precip_type_cts = np.argmax(np.stack(precip_type_cts), axis=0)
    precip_type_cts = np.nan_to_num(precip_type_cts, nan=-1).astype("int8")

    dataset_r = xr.Dataset({
        "latitude": (("latitude"), lats_out),
        "longitude": (("longitude"), lons_out),
        "surface_precip": (("latitude", "longitude"), surface_precip),
        "rqi": (("latitude", "longitude"), rqi),
        "precip_type": (("latitude", "longitude"), precip_type_cts),
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
        "scale_factor": 1 / 127,
        "dtype": "int8",
        "zlib": True,
        "_FillValue": -1
    }

    encoding["precip_type"] = {
        "dtype": "uint8",
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
        time_step=timedelta(minutes=15),
        include_scan_time=False
):
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
        include_scan_time: Ignored. Included for compatibility.
    """
    output_folder = Path(output_folder) / "mrms"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=24)
    time = start_time
    files = []
    while time < end_time:
        output_filename = get_output_filename(to_datetime64(time))
        time_range = TimeRange(time, time)
        if not (output_folder / output_filename).exists():
            files += zip(
                mrms.precip_rate.find_files(time_range)[-1:],
                mrms.radar_quality_index.find_files(time_range)[-1:],
                mrms.precip_flag.find_files(time_range)
            )
        time = time + time_step

    for precip_rate_file, rqi_file, precip_flag_file in files:
        with TemporaryDirectory() as tmp:
            try:
                precip_rate_file = precip_rate_file.download(destination=tmp)
                rqi_file = rqi_file.download(destination=tmp)
                precip_flag_file = precip_flag_file.download(destination=tmp)

                precip_rate_data = mrms.precip_rate.open(precip_rate_file)
                rqi_data = mrms.radar_quality_index.open(rqi_file)
                precip_flag_data = mrms.precip_flag.open(precip_flag_file)

                dataset = xr.merge(
                    [precip_rate_data, rqi_data, precip_flag_data],
                    compat="override"
                )
                dataset = dataset.rename({
                    "precip_rate": "surface_precip",
                    "radar_quality_index": "rqi",
                    "precip_flag": "precip_type"
                })
                dataset = resample_mrms_data(dataset)
                save_file(dataset, output_folder)
            except Exception:
                LOGGER.exception(
                    "The following error was encountered while processing "
                    " MRMS files (%s, %s)",
                    precip_rate_file,
                    rqi_file
                )
