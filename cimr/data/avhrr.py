"""
cimr.data.avhrr
===============

Functionality for reading and processing AHRR data.
"""
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pansat.roi import some_inside, find_overpasses
from pansat.products.satellite.avhrr import l1b_avhrr
from pansat.download.providers.eumetsat import EUMETSATProvider
from pyresample import geometry, kd_tree
from satpy.scene import Scene
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
        if name[0] == "S":
            date_1, date_2 = name.split("_")[5:7]
        else:
            date_1, date_2 = name.split("_")[4:6]

        date_1 = datetime.strptime(date_1[:-1], "%Y%m%d%H%M%S")
        date_2 = datetime.strptime(date_2[:-1], "%Y%m%d%H%M%S")
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
            file
            for file, date in zip(files, dates)
            if (date >= start_time) and (date <= end_time)
        ]

        return files

    def __init__(self, filename):
        """
        Create AVHRR file but don't load the data yet.

        Args:
            filename: Path to the file containing the AVHRR data.
        """
        self.filename = Path(filename)

    def __repr__(self):
        return f"AVHRRFile('{self.filename}')"

    def contains_roi(self):
        """
        Does the file contain observations over the nordic countries?
        """
        with xr.open_dataset(self.filename) as ds:
            return some_inside(ROI_NORDIC, ds)

    def _to_xarray_dataset_native(self):
        """
        Conversion to ``xarray.Dataset`` for data in native format.
        """

        dataset_names = ["1", "2", "3b", "4", "5"]
        new_names = {name: f"visir_{i + 1:02}" for i, name in enumerate(dataset_names)}

        with TemporaryDirectory() as tmp:

            # Decompress if necessary
            if self.filename.suffix == ".zip":
                args = ["unzip", self.filename.absolute()]
                subprocess.run(args, cwd=tmp)
                filename = (Path(tmp) / self.filename.name).with_suffix(".nat")
            else:
                filename = self.filename

            # Load datasets and coordinates
            scene = Scene([filename], reader="avhrr_l1b_eps")
            scene.load(["latitude", "longitude"] + dataset_names)
            lats = scene["latitude"].compute()
            lons = scene["longitude"].compute()
            dataset = scene.to_xarray_dataset().compute()
            dataset["latitude"] = lats
            dataset["longitude"] = lons

        dataset = dataset.rename(new_names).drop("crs")

        # Calculate scan time from start and end time.
        name = Path(self.filename).stem
        date_1, date_2 = name.split("_")[4:6]
        start_time = pd.Timestamp(
            datetime.strptime(date_1[:-1], "%Y%m%d%H%M%S")
        ).to_datetime64()
        end_time = pd.Timestamp(
            datetime.strptime(date_2[:-1], "%Y%m%d%H%M%S")
        ).to_datetime64()
        x = np.linspace(0, 1, dataset.y.size)
        time = np.interp(x, [0, 1], [np.float32(start_time), np.float32(end_time)])
        dataset["time"] = ("y", time.astype(start_time.dtype))

        return dataset

    def to_xarray_dataset(self):
        """
        Load data from file into xarray dataset.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        if self.filename.suffix != ".nc":
            return self._to_xarray_dataset_native()

        keys = ["lon", "lat", "image1", "image2", "image3", "image4", "image5"]
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
        swath, NORDIC_2, radius_of_influence=2e3, neighbours=1
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
            "nn",
            NORDIC_2.shape,
            data_in.data[0][mask],
            valid_in,
            valid_out,
            indices,
            fill_value=data_in.attrs["_FillValue"],
        )
        results[name_out] = (("y", "x"), data_out)

    results.attrs["time"] = str(dataset.attrs["time"])
    return results


def resample_scene(scene):
    """
    Resample scene to 2km grid over the nordic countries.

    Args:
        scene: ``xarray.Dataset`` containing the orbit data
            to resample to the NORDIC cimr domain at 2km
             resolution.

    Return:
        An ``xarray.Dataset`` containing the resampled
        observations and the mean observation time as
        attribute 'time'.
    """
    # Resample scene
    lons = scene.longitude.data
    lats = scene.latitude.data
    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    info = kd_tree.get_neighbour_info(
        swath, NORDIC_2, radius_of_influence=5e3, neighbours=1
    )
    valid_in, valid_out, indices, _ = info

    results = xr.Dataset()
    names = [f"visir_{i:02}" for i in range(1, 6)]
    for name in names:
        data_in = scene[name]
        data_out = kd_tree.get_sample_from_neighbour_info(
            "nn",
            NORDIC_2.shape,
            data_in.data,
            valid_in,
            valid_out,
            indices,
            fill_value=np.nan,
        )
        results[name] = (("y", "x"), data_out)
    results.attrs["time"] = scene.time.mean().data.item()
    return results


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            AVHRR observations.
        output_folder: The folder to which to write the training data.

    """
    comp = {"dtype": "int16", "scale_factor": 0.01, "zlib": True, "_FillValue": -99}
    encoding = {f"visir_{i:02}": comp for i in range(1, 6)}

    scenes = find_overpasses(ROI_NORDIC, dataset)
    for scene in scenes:

        scene = resample_scene(scene)

        # Make sure at least one pixel is valid.
        valid_pixels = 0
        for i in range(1, 6):
            valid_pixels += np.isfinite(scene[f"visir_{i:02}"].data).sum()
        if valid_pixels / 5 < 500:
            continue

        # Determine name of output file.
        time_15 = round_time(scene.attrs["time"])
        year = time_15.year
        month = time_15.month
        day = time_15.day
        hour = time_15.hour
        minute = time_15.minute
        filename = f"visir_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        output_filename = Path(output_folder) / filename

        if output_filename.exists():
            dataset_out = xr.load_dataset(output_filename)
            for i in range(1, 6):
                name = f"visir_{i:02}"
                var_in = scene[name]
                var_out = dataset_out[name]
                mask = ~np.isnan(var_in.data)
                var_out.data[mask] = var_in.data[mask]
            dataset_out.to_netcdf(output_filename, encoding=encoding)
        else:
            scene.to_netcdf(output_filename, encoding=encoding)


def process_day(year, month, day, output_folder, path=None):
    """
    Extract training data from a day of AVHRR observations.

    Args:
        year: The year
        month: The month
        day: The day
        output_folder: The folder to which to write the extracted
            observations.
        path: Not used, included for compatibility.
    """
    output_folder = Path(output_folder) / "visir"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    provider = EUMETSATProvider(l1b_avhrr)
    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(hours=23, minutes=59)
    bb = (ROI_NORDIC[0], ROI_NORDIC[1], ROI_NORDIC[2], ROI_NORDIC[3])
    files = provider.get_files_in_range(start_time, end_time, bounding_box=bb)

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for link in files:
            print("processing: ", link)
            filename = provider.download_file(link, tmp)
            data = AVHRR(filename).to_xarray_dataset()
            save_file(data, output_folder)
