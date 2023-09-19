"""
cimr.data.amsr2
===============

This module implements the interface to extract AMSR2 observations for
the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pansat.roi import find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l1c_gcomw1_amsr2
from pyresample import geometry, kd_tree
import xarray as xr

from cimr.utils import round_time
from cimr.data.resample import resample_tbs


def save_scene(time, tbs_r, output_folder, time_step):
    """
    Save training data scene.

    Args:
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        tbs_r: An xarray.Dataset containing the resampled brigthness
            temperatures.
        output_folder: The folder to which to write the training data.
    """
    minutes = time_step.seconds // 60
    time_15 = round_time(time, minutes=minutes)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"amsr2_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    tbs_r = tbs_r.transpose("channels", "y", "x", "swath_centers")

    comp = {
        "dtype": "int16",
        "scale_factor": 0.01,
        "zlib": True,
        "_FillValue": -99,
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
        year: int,
        month: int,
        day: int,
        output_folder: Path,
        path: Path = Optional[None],
        time_step: timedelta = timedelta(minutes=15),
        include_scan_time=False
) -> None:
    """
    Extract training data from a single AMSR2 L1C file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the L1C data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
        time_step: The time step between consecutive retrieval steps.
        include_scan_time: If set to 'True', the resampled scan time will
            be included in the extracted training input.
    """
    data = data.copy()
    data["latitude"] = data["latitude_s1"]
    data["longitude"] = data["longitude_s2"]

    scenes = find_overpasses(domain["roi"], data)
    for scene in scenes:
        tbs_r = resample_tbs(
            domain[8],
            data,
            radius_of_influence=10e3,
            n_swaths=6,
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
    Extract AMSR2 input observations for the CIMR retrieval.

    Args:
        domain: A domain dict specifying the area for which to
            extract AMSR2 input data.
        year: The year.
        month: The month.
        day: The day.
        output_folder: The root of the directory tree to which to write
            the training data.
        path: Not used, included for compatibility.
        time_step: The time step between consecutive retrieval steps.
        include_scan_time: If set to 'True', the resampled scan time will
            be included in the extracted training input.
    """
    output_folder = Path(output_folder) / "amsr2"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    provider = GesdiscProvider(l1c_gcomw1_amsr2)
    product_files = provider.get_files_in_range(start_time, end_time)
    # For all file on given day.
    for filename in product_files:
        # Check if swath covers ROI.
        swath = parse_swath(provider.download_metadata(filename))
        if swath.intersects(domain["roi_poly"].to_geometry()):
            # Extract observations
            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                provider.download_file(filename, tmp / filename)
                data = l1c_gcomw1_amsr2.open(tmp / filename)
                process_file(
                    domain,
                    data,
                    output_folder,
                    time_step,
                    include_scan_time=include_scan_time
                )
