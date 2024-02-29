"""
chimp.data.goes
===============

Defines CHIMP input datasets for observations from the GOES 16, 17, and 18 satellites.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import pandas as pd
from pansat.roi import any_inside
from pansat.time import TimeRange, to_datetime64
from pansat.download.providers import GOESAWSProvider
from pansat.products.satellite.goes import (
    GOES16L1BRadiances,
    GOES17L1BRadiances,
    GOES18L1BRadiances
)
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from chimp import areas
from chimp.utils import round_time
from chimp.data.input import InputDataset
from chimp.data.utils import get_output_filename

LOGGER = logging.getLogger(__name__)



CHANNEL_CONFIGURATIONS = {
    "ALL": list(range(1, 17)),
    "MATCHED": [
        2,   # 0.635 um
        3,   # 0.81 um
        5,   # 1.64 um
        7,   # 3.92 um
        8,   # 6.25 um
        10,  # 7.35 um
        11,  # 8.7 um
        12,  # 9.66 um
        13,  # 10.3 um
        15,  # 12 um
        16,  # 13.4 um
    ]
}


def download_and_resample_goes_data(
        time: datetime,
        domain: "Area",
        product_class: type,
        channel_configuration: str = "ALL"
):
    """
    Download GOES data closest to a given requested time step, resamples it
    and returns the result as an xarray dataset.

    Args:
        time: A datetime object specfiying the time for which to download the
            GOES observations.

    Return:
        An xarray dataset containing the GOES observations resampled to the
        CONUS_4 domain.
    """
    channels = CHANNEL_CONFIGURATIONS[channel_configuration]
    channel_names = [f"C{channel:02}" for channel in channels]

    goes_files = []

    for band in channels:
        prod = product_class("F", band)
        time_range = TimeRange(
            to_datetime64(time) - np.timedelta64(7 * 60, "s"),
            to_datetime64(time) + np.timedelta64(7 * 60, "s"),
        )
        recs = prod.get(time_range)

        if len(recs) == 0:
            return None

        file_inds = TimeRange(time, time).find_closest_ind(
            [rec.temporal_coverage for rec in recs]
        )
        goes_files.append(recs[file_inds[0]])

    scene = Scene([str(rec.local_path) for rec in goes_files], reader="abi_l1b")
    scene.load(channel_names)
    scene = scene.resample(domain)
    data = scene.to_xarray_dataset().compute()

    obs_refl = []
    obs_therm = []

    for channel in channels:
        ch_name = f"C{channel:02}"
        if channel <= 6:
            obs_refl.append(data[ch_name].data)
        else:
            obs_therm.append(data[ch_name].data)
    obs_refl = np.stack(obs_refl, -1)
    obs_therm = np.stack(obs_therm, -1)

    obs = xr.Dataset({
        "refls": (("y", "x", "channels_refl"), obs_refl),
        "tbs": (("y", "x", "channels_therm"), obs_therm)
    })
    start_time = data.attrs["start_time"]
    d_t = data.attrs["end_time"] - data.attrs["start_time"]
    obs.attrs["time"] = to_datetime64(start_time + 0.5 * d_t)
    return  obs


def save_file(dataset, output_folder, filename):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            GOES observations.
        output_folder: The folder to which to write the training data.

    """
    dataset = dataset.copy()
    output_filename = Path(output_folder) / filename
    dataset.attrs["time"] = str(dataset.attrs["time"])
    encoding = {}

    dataset["refls"].data[:] = np.minimum(dataset["refls"].data, 127)
    encoding["refls"] = {
        "dtype": "uint8",
        "_FillValue": 255,
        "scale_factor": 0.5,
        "zlib": True
    }
    dataset["tbs"].data[:] = np.clip(dataset["tbs"].data, 195, 323)
    encoding["tbs"] = {
        "dtype": "uint8",
        "scale_factor": 0.5,
        "add_offset": 195,
        "_FillValue": 255,
        "zlib": True
    }
    dataset.to_netcdf(output_filename, encoding=encoding)


class GOES(InputDataset):
    """
    Input data class for GOES observations.

    This class represents satellite observations from any of the recent GOES satellites.
    """
    def __init__(
            self, series: str, config: str, product_class: type
    ):
        self.series = series
        self.config = config
        self.product_class = product_class
        if config.upper() == "ALL":
            dataset_name = "goes_" + series.lower()
            input_name = "goes"
        else:
            dataset_name = "goes_" + series.lower() + "_" + config.lower()
            input_name = "goes_" + config.lower()
        super().__init__(dataset_name, input_name, 4, ["refls", "tbs"])
        self.n_channels = 11

    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(minutes=15),
            include_scan_time=False
    ):
        """
        Extract training data from a day of GOES observations.

        Args:
            domain: A domain object identifying the spatial domain for which
                to extract input data.
            year: The year
            month: The month
            day: The day
            output_folder: The folder to which to write the extracted
                observations.
            path: Not used, included for compatibility.
            time_step: The temporal resolution of the training data.
            include_scan_time: Not used.
        """
        output_folder = Path(output_folder) / self.dataset_name
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
        time = start_time
        while time < end_time:
            dataset = download_and_resample_goes_data(
                time,
                domain[self.scale],
                self.product_class,
                self.config
            )
            filename = get_output_filename(self.input_name, time, minutes=time_step)
            save_file(dataset, output_folder, filename)
            time = time + time_step


goes_16 = GOES("16", "ALL", GOES16L1BRadiances)
goes_17 = GOES("17", "ALL", GOES17L1BRadiances)
goes_18 = GOES("18", "ALL", GOES18L1BRadiances)
goes_16_matched = GOES("16", "MATCHED", GOES16L1BRadiances)
goes_17_matched = GOES("17", "MATCHED", GOES17L1BRadiances)
goes_18_matched = GOES("18", "MATCHED", GOES18L1BRadiances)
