"""
cimr.data.atms
==============

This module implements the interface to extract MHS observations for
the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from pansat.roi import find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import (
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa19_mhs,
)
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.utils import round_time
from cimr.data.utils import make_microwave_band_array
from cimr.definitions import N_CHANS


# The pansat products providing access to MHS L1C data.
MHS_PRODUCTS = [
    l1c_noaa19_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
]


BANDS = {
    "mw_90": {
        1: 0,  # 89 GHz, V
    },
    "mw_160": {
        1: 1,  # 166 GHz, H
    },
    "mw_183": {
        0: 2,  # 183 +/- 1
        2: 3,  # 183 +/- 3
        4: 4,  # 183 +/- 7
    },
}


def resample_swaths(domain, scene):
    """
    Resample MHS observations to 8 kilometer resolutions.

    Args:
        domain: A domain dict describing the domain over which to extract
            training data.
        scene: An xarray.Dataset containing the observations to
            resample.

    Return:
        The resampled observaitons.
    """
    lons = scene["longitude"].data
    lats = scene["latitude"].data
    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    tbs = scene[f"tbs"].data
    tbs_r = kd_tree.resample_nearest(
        swath, tbs, domain[8], radius_of_influence=16e3, fill_value=np.nan
    )
    return tbs_r


def save_scene(domain, scene, tbs_r, output_folder):
    """
    Save training data scene.

    Args:
        domain: The domain to which the observations are mapped.
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        tbs_r: A dict containing the resampled swaths.
        output_folder: The folder to which to write the training data.
    """
    # Only use scenes with substantial information
    if np.any(np.isfinite(tbs_r[..., 0]), axis=-1).sum() < 100:
        return None

    # Extract observations and map to CIMR bands.
    results = {}
    for band, chans in BANDS.items():
        obs = make_microwave_band_array(domain, band)
        for ind_out, ind_in in chans.items():
            obs[..., ind_out] = tbs_r[..., ind_in]
        results[band] = obs

    dataset = xr.Dataset()
    for band in BANDS:
        if band == "mw_low":
            dims = ("y_16", "x_16", f"channels_{band}")
        else:
            dims = ("y_8", "x_8", f"channels_{band}")
        dataset[band] = (dims, results[band].data)

    time = scene.scan_time.mean().item()
    time_15 = round_time(time)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"mw_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    if output_filename.exists():
        dataset_out = xr.load_dataset(output_filename)
        for band in BANDS:
            if band in dataset_out:
                var_out = dataset_out[band]
                var_in = dataset[band]
                missing = ~np.isfinite(var_in.data)
                var_out.data[~missing] = var_in.data[~missing]
            else:
                dataset_out[band] = dataset[band]
        dataset_out.attrs["sensor"] += ", " + scene.attrs["InstrumentName"]
        dataset_out.attrs["satellite"] += ", " + scene.attrs["SatelliteName"]
        dataset_out.to_netcdf(output_filename)
    else:
        comp = {
            "dtype": "int16",
            "scale_factor": 0.01,
            "zlib": True,
            "_FillValue": -99,
        }
        dataset.attrs["sensor"] = scene.attrs["InstrumentName"]
        dataset.attrs["satellite"] = scene.attrs["SatelliteName"]
        encoding = {band: comp for band in BANDS}
        dataset.to_netcdf(output_filename, encoding=encoding)

    return None


def process_file(domain, data, output_folder):
    """
    Extract training data from a single ATMS L1C file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the L1C data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
    """
    data = data.copy()
    scenes = find_overpasses(domain["roi"], data)

    for scene in scenes:
        tbs_r = resample_swaths(domain, data)
        save_scene(domain, scene, tbs_r, output_folder)


def process_day(domain, year, month, day, output_folder, path=None):
    """
    Extract training data for a day of MHS observations.

    Args:
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
    """
    output_folder = Path(output_folder) / "microwave"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    print(year, month, day)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    # Iterate over platform.
    for product in MHS_PRODUCTS:
        provider = GesdiscProvider(product)
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
                    data = product.open(tmp / filename)
                    process_file(domain, data, output_folder)
