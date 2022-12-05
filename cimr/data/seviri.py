"""
cimr.data.seviri
================

Functionality for reading and processing SEVIRI data.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside
from pansat.download.providers.eumetsat import EUMETSATProvider
from pansat.products.satellite.meteosat import l1b_msg_seviri
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from cimr.areas import NORDIC_4
from cimr.utils import round_time


LOGGER = logging.getLogger(__name__)


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
            file
            for file, date in zip(files, dates)
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
            filename = str((Path(tmp) / self.filename.name).with_suffix(".nat"))
            scene = Scene([filename], reader="seviri_l1b_native")

            datasets = scene.available_dataset_names()
            datasets = np.array([name for name in datasets if name != "HRV"])
            wavelengths = np.array([int(name[-3:]) for name in datasets])
            datasets = list(datasets[np.argsort(wavelengths)])
            names = {name: f"geo_{i + 1:02}" for i, name in enumerate(datasets)}
            scene.load(datasets)
            scene_r = scene.resample(NORDIC_4, radius_of_influence=32e3, fill_value=np.nan)

        dataset = scene_r.to_xarray_dataset().compute().rename(names)
        for var in dataset.variables:
            dataset[var].attrs = {}

        dataset.attrs = {}
        dataset.attrs["time"] = str(SEVIRI.filename_to_date(self.filename))
        return dataset.drop("crs")


def get_output_filename(time):
    """
    Get filename for training sample.

    Args:
        time: The observation time.

    Return:
        A string specifying the filename of the training sample.
    """
    time_15 = round_time(time)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"seviri_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    return filename




def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            SEVIRI observations.
        output_folder: The folder to which to write the training data.

    """
    filename = get_output_filename(dataset.attrs["time"])
    output_filename = Path(output_folder) / filename

    comp = {"dtype": "int16", "scale_factor": 0.01, "zlib": True, "_FillValue": -99}
    encoding = {f"geo_{i:02}": comp for i in range(1, 12)}
    dataset.to_netcdf(output_filename, encoding=encoding)


def process_day(year, month, day, output_folder, path=None):
    """
    Extract training data from a day of SEVIRI observations.

    Args:
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
    """
    output_folder = Path(output_folder) / "geo"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    provider = EUMETSATProvider(l1b_msg_seviri)
    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    files = provider.get_files_in_range(start_time, end_time)
    existing_files = [
        f.name for f in output_folder.glob(f"seviri_{year}{month:02}{day:02}*.nc")
    ]

    print(len(existing_files), len(files))
    # Nothing to do if all files are already available.
    if len(existing_files) == len(files):
        return None

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for link in files:

            try:
                filename = link.split("/")[-1]
                time = SEVIRI.filename_to_date(filename)
                filename = get_output_filename(time)

                if filename in existing_files:
                    continue

                filename = provider.download_file(link, tmp)
                data = SEVIRI(filename).to_xarray_dataset()
                save_file(data, output_folder)

            except Exception as e:
                LOGGER.error(
                    "Processing of input_file '%s' failed with the "
                    "following error: %s",
                    link,
                    e
                )
