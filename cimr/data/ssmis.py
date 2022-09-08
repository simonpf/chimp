"""
cimr.data.ssmis
===============

Functionality for reading and processing SSMIS data.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside, find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import (
    l1c_f17_ssmis,
    l1c_f18_ssmis
)
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.areas import ROI_NORDIC, NORDIC_8, ROI_POLY
from cimr.utils import round_time
from cimr.definitions import N_CHANS

BANDS = {
    "mw_90": [(10, 0), (9, 1)],
    "mw_160": [(5, 0)],
    "mw_183": [(6, 0), (7, 2), (8, 4)]
}

def make_band(band):
    n_chans = N_CHANS[band]
    return xr.DataArray(np.zeros(NORDIC_8.shape + (n_chans,), dtype=np.float32))


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing MHS L1C data.
        output_folder: The folder to which to write the training data.
    """
    dataset = dataset.copy()
    dataset["latitude"] = dataset["latitude_s1"]
    dataset["longitude"] = dataset["longitude_s2"]

    scenes = find_overpasses(ROI_NORDIC, dataset)
    for scene in scenes:
        time = scene.scan_time.mean().item()
        time_15 = round_time(time)
        year = time_15.year
        month = time_15.month
        day = time_15.day
        hour = time_15.hour
        minute = time_15.minute

        tbs_r = []
        for i in range(4):
            suffix = f"_s{i + 1}"
            lons = scene[f"longitude{suffix}"].data
            lats = scene[f"latitude{suffix}"].data
            swath = geometry.SwathDefinition(lons=lons, lats=lats)
            tbs = scene[f"tbs{suffix}"].data
            tbs_r.append(kd_tree.resample_nearest(
                swath, tbs, NORDIC_8, radius_of_influence=64e3, fill_value=np.nan
            ))
        tbs_r = np.concatenate(tbs_r, axis=-1)
        assert tbs_r.shape[-1] == 11

        # Only use scenes with substantial information
        if np.any(np.isfinite(tbs_r[..., 0]), axis=-1).sum() < 100:
            continue

        results = {}
        for band, chans in BANDS.items():
            b = make_band(band)
            for ind_in, ind_out in chans:
                b.data[..., ind_out] = tbs_r[..., ind_in]
            results[band] = b

        results = xr.Dataset(
            {
                band: (("y", "x", f"channels_{band}"), results[band].data)
                for band in BANDS
            }
        )

        filename = f"mw_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        output_filename = Path(output_folder) / filename

        if output_filename.exists():
            dataset_out = xr.load_dataset(output_filename)
            for band in BANDS.keys():
                if band in dataset_out:
                    var_out = dataset_out[band]
                    var_in = results[band]
                    missing = ~np.isfinite(var_in.data)
                    var_out.data[~missing] = var_in.data[~missing]
                else:
                    dataset_out[band] = results[band]
            dataset_out.attrs["sensor"] = dataset.attrs["InstrumentName"]
            dataset_out.attrs["satellite"] = dataset.attrs["SatelliteName"]
            dataset_out.to_netcdf(output_filename)
        else:
            comp = {
                "dtype": "int16",
                "scale_factor": 0.01,
                "zlib": True,
                "_FillValue": -99,
            }
            results.attrs["sensor"] = dataset.attrs["InstrumentName"]
            results.attrs["satellite"] = dataset.attrs["SatelliteName"]
            encoding = {band: comp for band in BANDS.keys()}
            results.to_netcdf(output_filename, encoding=encoding)


SSMIS_PRODUCTS = [
    l1c_f17_ssmis,
    l1c_f18_ssmis,
]


def process_day(year, month, day, output_folder, path=None):
    """
    Extract training data for a day of SSMIS observations.

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

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    files = []
    # Iterate over platform.
    for product in SSMIS_PRODUCTS:
        provider = GesdiscProvider(product)
        product_files = provider.get_files_in_range(start_time, end_time)
        print(start_time, end_time, product_files)
        # For all file on given day.
        for filename in product_files:
            # Check if swath covers ROI.
            print(filename)
            swath = parse_swath(provider.download_metadata(filename))
            if swath.intersects(ROI_POLY.to_geometry()):
                # Extract observations
                with TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    print("processing: ", filename)
                    provider.download_file(filename, tmp / filename)
                    data = product.open(tmp / filename)
                    save_file(data, output_folder)
