"""
cimr.data.mhs
=============

Functionality for reading and processing MHS data.
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
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
)
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.areas import ROI_NORDIC, NORDIC_8, ROI_POLY
from cimr.utils import round_time
from cimr.definitions import N_CHANS


class MHS:
    """
    Interface class to read MHS data.
    """

    @staticmethod
    def filename_to_date(filename):
        """
        Extract time from filename.

        Args:
            filename: The name or path to the file.

        Return
            ``np.datetime64`` object representing the time.
        """
        name = Path(filename).stem
        date = name.split("_")[4]
        date = datetime.strptime(date, "%Y%m%d%H%M%S")
        return pd.Timestamp(date).to_datetime64()

    @staticmethod
    def find_files(base_dir, start_time=None, end_time=None):
        """
        Find MHS files.

        Args:
            base_dir: Root directory to search through.
            start_time: Optional start time to limit the search.
            end_time: Optional end time to limit the search.

        Return:
            A list of the found files.
        """
        files = list(Path(base_dir).glob("**/W_XX-EUMETSAT-Darmstadt*+MHS_*.nc"))
        dates = np.array(list((map(MHS.filename_to_date, files))))

        if len(dates) == 0:
            return []

        if start_time is None:
            start_time = dates.min()
        else:
            start_time = np.datetime64(start_time)

        if end_time is None:
            end_time = dates.max()
        else:
            end_time = np.datetime64(end_time)

        return [
            file
            for file, date in zip(files, dates)
            if (date >= start_time) and (date <= end_time)
        ]

        return files

    def __init__(self, filename):
        """
        Create MHS file but don't load the data yet.

        Args:
            filename: Path to the file containing the MHS data.
        """
        self.filename = filename

    def contains_roi(self):
        """
        Does the file contain observations over the nordic countries?
        """
        with xr.open_dataset(self.filename) as ds:
            return any_inside(ROI_NORDIC, ds)

    def to_xarray_dataset(self):
        """
        Load data from file into xarray dataset.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        data = xr.load_dataset(self.filename, decode_times=False)

        start_time = data["record_start_time"].data
        start_time = np.datetime64("2000-01-01T00:00:00") + start_time.astype(
            "timedelta64[s]"
        )
        end_time = data["record_stop_time"].data
        end_time = np.datetime64("2000-01-01T00:00:00") + end_time.astype(
            "timedelta64[s]"
        )

        time = start_time + 0.5 * (end_time - start_time)

        data["time"] = (("along_track",), time)

        names = [f"channel_{i:01}" for i in range(1, 6)]
        names += ["lon", "lat", "time"]
        data = data[names]

        new_names = {f"channel_{i:01}": f"channel_{i:02}" for i in range(1, 6)}
        new_names["lon"] = "longitude"
        new_names["lat"] = "latitude"
        data = data.rename(new_names)

        return data


BANDS = {"mw_90": [(0, 1)], "mw_160": [(1, 1)], "mw_183": [(2, 0), (3, 2), (4, 4)]}


def make_band(band):
    n_chans = N_CHANS[band]
    return xr.DataArray(np.zeros(NORDIC_8.shape + (n_chans,), dtype=np.float32))


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            MHS observations.
        output_folder: The folder to which to write the training data.

    """
    scenes = find_overpasses(ROI_NORDIC, dataset)
    for scene in scenes:
        time = scene.scan_time.mean().item()
        time_15 = round_time(time)
        year = time_15.year
        month = time_15.month
        day = time_15.day
        hour = time_15.hour
        minute = time_15.minute

        lons = scene.longitude.data
        lats = scene.latitude.data

        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        tbs = scene.tbs.data
        tbs_r = kd_tree.resample_nearest(
            swath, tbs, NORDIC_8, radius_of_influence=64e3, fill_value=np.nan
        )
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


MHS_PRODUCTS = [
    l1c_noaa19_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
]


def process_day(year, month, day, output_folder, path=None):
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
    files = []
    # Iterate over platform.
    for product in MHS_PRODUCTS:
        provider = GesdiscProvider(product)
        product_files = provider.get_files_in_range(start_time, end_time)
        # For all file on given day.
        for filename in product_files:
            # Check if swath covers ROI.
            swath = parse_swath(provider.download_metadata(filename))
            if swath.intersects(ROI_POLY.to_geometry()):
                # Extract observations
                with TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    provider.download_file(filename, tmp / filename)
                    data = product.open(tmp / filename)
                    save_file(data, output_folder)
