"""
cimr.data.mhs
=============

Functionality for reading and processing MHS data.
"""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside, find_overpasses
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.areas import ROI_NORDIC, NORDIC_8
from cimr.utils import round_time

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
            file for file, date in zip(files, dates)
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
        start_time = (np.datetime64("2000-01-01T00:00:00") +
                      start_time.astype("timedelta64[s]"))
        end_time = data["record_stop_time"].data
        end_time = (np.datetime64("2000-01-01T00:00:00") +
                    end_time.astype("timedelta64[s]"))


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
        time = scene.time.mean().item()
        time_15 = round_time(time)
        year = time_15.year
        month = time_15.month
        day = time_15.day
        hour = time_15.hour
        minute = time_15.minute

        lons = scene.longitude.data
        lats = scene.latitude.data

        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        info = kd_tree.get_neighbour_info(
            swath,
            NORDIC_8,
            radius_of_influence=32e3,
            neighbours=1
        )
        valid_in, valid_out, indices, _ = info

        names = [f"channel_{i:02}" for i in range(1, 6)]

        results = xr.Dataset()
        for name in names:

            data_in = scene[name]
            data_out = kd_tree.get_sample_from_neighbour_info(
                'nn',
                NORDIC_8.shape,
                data_in.data,
                valid_in,
                valid_out,
                indices,
                fill_value=np.nan
            )
            results[name] = (("y", "x"), data_out)
            results[name].attrs = data_in.attrs

        filename = f"mhs_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        output_filename = Path(output_folder) / filename

        print(output_filename)

        if output_filename.exists():
            dataset_out = xr.load_dataset(output_filename)
            for i in range(1, 6):
                name = f"channel_{i:02}"
                var_out = dataset_out[name]
                var_in = results[name]
                missing = ~np.isfinite(var_in.data)
                mask = var_in.data != missing
                var_out.data[mask] = var_in.data[mask]
        else:
            results.to_netcdf(output_filename)
