"""
cimr.data.seviri
================

Functionality for reading and processing SEVIRI data.
"""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.areas import NORDIC_4
from cimr.utils import round_time


class SEVIRI:
    """
    Interface class to read SEVIRI data.
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
        date = name.split("-")[5][:14]
        date = datetime.strptime(date, "%Y%m%d%H%M%S")
        return pd.Timestamp(date).to_datetime64()

    @staticmethod
    def find_files(base_dir, start_time=None, end_time=None):
        """
        Find SEVIRI files.

        Args:
            base_dir: Root directory to search through.
            start_time: Optional start time to limit the search.
            end_time: Optional end time to limit the search.

        Return:
            A list of the found files.
        """
        files = list(Path(base_dir).glob("**/MSG4-SEVI*.zip"))
        dates = np.array(list((map(SEVIRI.filename_to_date, files))))

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
        Create SEVIRI file object but don't load the data yet.

        Args:
            filename: Path to the file containing the SEVIRI data.
        """
        self.filename = filename

    def to_xarray_dataset(self):
        """
        Load data from file into xarray dataset.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        with TemporaryDirectory() as tmp:

            args = ["unzip", self.filename]
            subprocess.run(args, cwd=tmp)
            filename = (Path(tmp) / self.filename.name).with_suffix(".nat")
            scene = Scene([filename], reader="seviri_l1b_native")

            datasets = scene.all_dataset_names()
            print(datasets)
            scene.load(datasets)

            datasets = np.array([name for name in datasets if name != "HRV"])
            wavelengths = np.array([int(name[-3:]) for name in datasets])
            datasets = ["HRV"] + list(datasets[np.argsort(wavelengths)])
            names = {name: f"channel_{i + 1:02}" for i, name in enumerate(datasets)}

            scene.load(datasets)
            scene_r = scene.resample(NORDIC_4, radius_of_influence=16e3)

        dataset = scene_r.to_xarray_dataset().compute().rename(names)
        for var in dataset.variables:
            dataset[var].attrs = {}

        dataset.attrs = {}
        dataset.attrs["time"] = str(SEVIRI.filename_to_date(self.filename))
        return dataset.drop("crs")


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            SEVIRI observations.
        output_folder: The folder to which to write the training data.

    """
    time_15 = round_time(dataset.attrs["time"])
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"seviri_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    encoding = {
        f"channel_{i:02}": {
            "dtype": "int16",
            "scale_factor": 0.1
        } for i in range(1, 13)
    }
    dataset.to_netcdf(output_filename, encoding=encoding)
