"""
cimr.data.atms
==============

This module implements the interface to extract ATMS observations for
the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from pansat.roi import find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import (
    l1c_noaa20_atms,
    l1c_npp_atms,
)
from pyresample import geometry, kd_tree
import xarray as xr

from cimr.utils import round_time
from cimr.data.utils import make_microwave_band_array
from cimr.data.resample import resample_tbs


# The pansat products providing access to ATMS L1C data.
ATMS_PRODUCTS = [
    l1c_noaa20_atms,
    l1c_npp_atms,
]


def save_scene(
        time: timedelta,
        tbs_r: xr.Dataset,
        output_folder: Path,
        time_step: timedelta
) -> None:
    """
    Save training data scene.

    Args:
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        tbs_r: A dict containing the resampled swaths.
        output_folder: The folder to which to write the training data.
        time_step: A 'datetime.timedelta' object specifying the fundamental
            time step of the retrieval.
    """
    minutes = time_step.seconds // 60
    time_15 = round_time(time, minutes=minutes)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"atms_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    tbs_r = tbs_r.transpose("channels", "y", "x", "swath_centers")

    comp = {
        "dtype": "uint16",
        "scale_factor": 0.01,
        "zlib": True,
        "_FillValue": 2 ** 16 - 1,
    }
    encoding = {
        "tbs": comp,
        "swath_center_col_inds": {"dtype": "int16"},
        "swath_center_row_inds": {"dtype": "int16"}
    }
    tbs_r.to_netcdf(output_filename, encoding=encoding)
    return None


def process_file(
        domain: dict,
        data: xr.Dataset,
        output_folder : Path,
        time_step: timedelta,
        include_scan_time: bool = False
) -> None:
    """
    Extract training data from a single ATMS L1C file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the L1C data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
        time_step: The time step used for the discretization of the input
            data.
        include_scan_time: Boolean flag indicating whether or not to include
            the resampled scan time in the extracted retrieval input.
    """
    data = data.copy()
    data["latitude"] = data["latitude_s1"]
    data["longitude"] = data["longitude_s2"]

    scenes = find_overpasses(domain["roi"], data)

    for scene in scenes:
        tbs_r = resample_tbs(
            domain[16],
            data,
            n_swaths=4,
            radius_of_influence=64e3,
            include_scan_time=include_scan_time
        )
        time = scene.scan_time.mean().data.item()
        tbs_r.attrs = scene.attrs
        save_scene(time, tbs_r, output_folder, time_step)


def process_day(
        domain: dict,
        year: int,
        month: int,
        day: int,
        output_folder: Path,
        path: Path = Optional[None],
        time_step: timedelta = timedelta(minutes=15),
        include_scan_time=False
) -> None:
    """
    Extract training data for a day of ATMS observations.

    Args:
        domain: A domain dict specifying the area for which to
            extract ATMS input data.
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
        time_step: A datetime.timedelta object specifying the time step
            of the retrieval.
        include_scan_time: If set to 'True', the resampled scan time will
            be included in the extracted training input.
    """
    output_folder = Path(output_folder) / "atms"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    # Iterate over platform.
    for product in ATMS_PRODUCTS:
        provider = GesdiscProvider(product)
        product_files = provider.get_files_in_range(start_time, end_time)
        # For all file on given day.
        for filename in product_files:
            # Check if swath covers ROI.
            swath = parse_swath(provider.download_metadata(filename))
            if swath.intersects(domain["roi_poly"].to_geometry()):
                # Extract observationsd
                with TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    provider.download_file(filename, tmp / filename)
                    data = product.open(tmp / filename)
                    process_file(
                        domain,
                        data,
                        output_folder,
                        time_step,
                        include_scan_time=include_scan_time
                    )
