"""
cimr.data.avhrr
===============

Functionality for reading and processing AHRR data.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pansat.roi import some_inside
from pyresample import geometry, kd_tree
import xarray as xr

from cimr.areas import ROI_NORDIC, NORDIC_2
from cimr.utils import round_time

class AVHRR:
    """
    Interface class to read AVHRR data.
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
        date_1, date_2 = name.split("_")[5:7]
        date_1 = datetime.strptime(date_1[:-2], "%Y%m%dT%H%M%S")
        date_2 = datetime.strptime(date_2[:-2], "%Y%m%dT%H%M%S")
        dt = date_2 - date_1
        return pd.Timestamp(date_1 + 0.5 * dt).to_datetime64()

    @staticmethod
    def find_files(base_dir, start_time=None, end_time=None):
        """
        Find AVHRR files.

        Args:
            base_dir: Root directory to search through.
            start_time: Optional start time to limit the search.
            end_time: Optional end time to limit the search.

        Return:
            A list of the found files.
        """
        files = list(Path(base_dir).glob("**/S_NWC_avhrr_*.nc"))
        dates = np.array(list((map(AVHRR.filename_to_date, files))))

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
        Create AVHRR file but don't load the data yet.

        Args:
            filename: Path to the file containing the AVHRR data.
        """
        self.filename = filename

    def contains_roi(self):
        """
        Does the file contain observations over the nordic countries?
        """
        with xr.open_dataset(self.filename) as ds:
            return some_inside(ROI_NORDIC, ds)

    def to_xarray_dataset(self):
        """
        Load data from file into xarray dataset.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        keys = [
            "lon", "lat", "image1", "image2", "image3",
            "image4", "image5"
        ]
        time = AVHRR.filename_to_date(self.filename)
        ds = xr.load_dataset(self.filename, decode_cf=False)[keys]
        ds.attrs["time"] = time
        return ds


def resample_to_area(dataset):
    """
    Resample observations to Scandinavia.

    Args:
        dataset: Data from a AVHRR file as ``xarray.Dataset``

    Return:
        An ``xarray.Dataset`` containing the observations resampled
        to Scandinavia.
    """
    missing = dataset.image5.attrs["_FillValue"]
    mask = dataset.image5.data[0] != missing

    lons = dataset.lon.data[mask]
    lats = dataset.lat.data[mask]

    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    info = kd_tree.get_neighbour_info(
        swath,
        NORDIC_2,
        radius_of_influence=2e3,
        neighbours=1
    )
    valid_in, valid_out, indices, _ = info

    names = [
        ("image1", "channel_1"),
        ("image2", "channel_2"),
        ("image3", "channel_3"),
        ("image4", "channel_4"),
        ("image5", "channel_5"),
    ]

    results = xr.Dataset()
    for name_in, name_out in names:

        data_in = dataset[name_in]
        data_out = kd_tree.get_sample_from_neighbour_info(
            'nn',
            NORDIC_2.shape,
            data_in.data[0][mask],
            valid_in,
            valid_out,
            indices,
            fill_value=data_in.attrs["_FillValue"]
        )
        results[name_out] = (("y", "x"), data_out)
        results[name_out].attrs = data_in.attrs

    results.attrs["time"] = str(dataset.attrs["time"])
    return results


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            AVHRR observations.
        output_folder: The folder to which to write the training data.

    """
    time_15 = round_time(dataset.attrs["time"])
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"avhrr_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    if output_filename.exists():
        dataset_out = xr.load_dataset(output_filename, decode_cf=False)
        for i in range(1, 6):
            name = f"channel_{i}"
            var_out = dataset_out[name]
            var_in = dataset[name]
            missing = var_in.attrs["_FillValue"]
            mask = var_in.data != missing
            var_out.data[mask] = var_in.data[mask]
    else:
        dataset.to_netcdf(output_filename)
