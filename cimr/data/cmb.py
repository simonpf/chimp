"""
cimr.data.cmb
=============

This module implements the interface to extract GPM CMB retrieval results
 for the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from pansat.metadata import parse_swath
from pansat.roi import find_overpasses
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l2b_gpm_cmb

from cimr.utils import round_time
from cimr.data.resample import resample_retrieval_targets


def save_scene(time, data_r, output_folder, time_step):
    """
    Save training data scene.

    Args:
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        data_r: An xarray.Dataset containing the resampled GPM CMB data.
        output_folder: The folder to which to write the training data.
    """
    minutes = time_step.seconds // 60
    time_15 = round_time(time, minutes=minutes)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"cmb_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    encoding = {
        "swath_center_col_inds": {"dtype": "int16"},
        "swath_center_row_inds": {"dtype": "int16"}
    }
    for var in data_r:
        encoding[var] = {"dtype": "float32", "zlib": True}
    data_r.to_netcdf(output_filename, encoding=encoding)
    return None



def process_file(domain, data, output_folder, time_step):
    """
    Extract training data from a single GPM CMB file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the GPM CMB data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
        time_step: The time step between consecutive retrieval steps.
    """
    data = data.copy()

    scenes = find_overpasses(domain["roi"], data)

    for scene in scenes:
        data_r = resample_retrieval_targets(
            domain[4],
            data,
            ["near_surf_precip_tot_rate"],
            radius_of_influence=5e3
        )
        time = scene.scan_time.mean().data.item()
        data_r.attrs = scene.attrs
        save_scene(time, data_r, output_folder, time_step)


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
    Extract DPR retrieval results for the CIMR retrieval.

    Args:
        domain: A domain dict specifying the area for which to
            extract CMB input data.
        year: The year.
        month: The month.
        day: The day.
        output_folder: The root of the directory tree to which to write
            the training data.
        path: Not used, included for compatibility.
        time_step: The time step between consecutive retrieval steps.
    """
    output_folder = Path(output_folder) / "cmb"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    provider = GesdiscProvider(l2b_gpm_cmb)

    product_files = provider.get_files_in_range(start_time, end_time)
    # For all file on given day.
    for filename in product_files:
        # Check if swath covers ROI.
        print(filename)
        swath = parse_swath(provider.download_metadata(filename))
        if swath.intersects(domain["roi_poly"].to_geometry()):
            # Extract observations
            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                provider.download_file(filename, tmp / filename)
                data = l2b_gpm_cmb.open(tmp / filename)
                process_file(domain, data, output_folder, time_step)
