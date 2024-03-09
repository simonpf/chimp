"""
chimp.data.goes
===============

Defines CHIMP input datasets for observations from the GOES 16, 17, and 18 satellites.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
import subprocess

import numpy as np
import pandas as pd
from pansat import FileRecord, TimeRange
from pansat.geometry import Geometry
from pansat.time import to_datetime64
from pansat.download.providers import GOESAWSProvider
from pansat.products.satellite.goes import (
    GOES16L1BRadiances,
    GOES17L1BRadiances,
    GOES18L1BRadiances
)
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.utils import round_time, get_output_filename

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


def get_time_range(recs: List[FileRecord]) -> TimeRange:
    """
    Determine time range covering a list of GOES file.

    Args:
        recs: A list of pansat FileRecord object pointing to the GOES data files.

    Return:
        A pansat.TimeRange object covering all time ranges of the given GOES
        files.
    """
    min_time = None
    max_time = None
    for rec in recs:
        if min_time is None:
            min_time = rec.start_time
        else:
            min_time = min(min_time, rec.start_time)
        if max_time is None:
            max_time = rec.start_time
        else:
            max_time = max(max_Time, rec.end_time)
    return TimeRange(min_time, max_time)

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
    scene.load(channel_names, generate=False)
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
        self.scale = 4
        self.product_class = product_class
        if config.upper() == "ALL":
            dataset_name = "goes_" + series.lower()
            input_name = "goes"
        else:
            dataset_name = "goes_" + series.lower() + "_" + config.lower()
            input_name = "goes_" + config.lower()
        super().__init__(dataset_name, input_name, 4, ["refls", "tbs"])
        self.n_channels = len(CHANNEL_CONFIGURATIONS[config])

        channels = CHANNEL_CONFIGURATIONS[config]
        channel_names = [f"C{channel:02}" for channel in channels]
        self.products = [product_class("F", channel) for channel in channels]

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ):
        """
        Find input data files within a given file range from which to extract
        training data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        found_files = []

        if path is not None:
            all_files = sorted(list(path.glob("**/*.nc")))

        for prod in self.products:
            if path is not None:
                recs = [
                    FileRecord.from_local(prod, path) for path in all_files
                    if prod.matches(path)
                ]
            else:
                recs = prod.find_files(TimeRange(start_time, end_time))

            matched_recs = {}
            matched_deltas = {}

            for rec in recs:
                tr_rec = rec.temporal_coverage
                time_c = to_datetime64(tr_rec.start + 0.5 * (tr_rec.end - tr_rec.start))
                time_n = round_time(time_c, time_step)
                delta = abs(time_c - time_n)

                min_delta = matched_deltas.get(time_n)
                if min_delta is None:
                    matched_recs[time_n] = rec
                    matched_deltas[time_n] = delta
                else:
                    if delta < min_delta:
                        matched_recs[time_n] = rec
                        matched_deltas[time_n] = delta

            found_files += list(matched_recs.values())

        return [rec.get().local_path for rec in found_files]


    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        product = [prod for prod in self.products if prod.matches(path)][0]
        channel = product.channel
        channel_name = f"C{channel:02}"
        scene = Scene([str(path)], reader="abi_l1b")
        scene.load([channel_name], generate=False)
        scene = scene.resample(domain[self.scale])
        data = scene.to_xarray_dataset().compute()

        obs_refl = []
        obs_therm = []

        time_range = product.get_temporal_coverage(path)
        time_c = time_range.start + 0.5 * (time_range.end - time_range.start)
        filename = get_output_filename(
            self.name, time_c, time_step
        )
        output_file = output_folder / filename

        if output_file.exists():
            dataset = xr.load_dataset(output_file)
        else:
            obs_refl = np.nan * np.zeros(domain[self.scale].shape + (6,), dtype=np.float32)
            obs_therm = np.nan * np.zeros(domain[self.scale].shape + (10,), dtype=np.float32)
            dataset = xr.Dataset({
                "refls": (("y", "x", "channels_refl"), obs_refl),
                "tbs": (("y", "x", "channels_therm"), obs_therm)
            })

        if channel <= 6:
            dataset["refls"].data[..., channel - 1] = data[channel_name]
        else:
            dataset["tbs"].data[..., channel - 1 - 6] = data[channel_name]

        dataset = dataset.copy()
        output_filename = Path(output_folder) / filename
        if "input_files" in dataset.attrs:
            dataset.attrs["input_files"] += ", " + path.name
        else:
            dataset.attrs["input_files"] = path.name
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


GOES_16 = GOES("16", "ALL", GOES16L1BRadiances)
GOES_17 = GOES("17", "ALL", GOES17L1BRadiances)
GOES_18 = GOES("18", "ALL", GOES18L1BRadiances)
GOES_16_MATCHED = GOES("16", "MATCHED", GOES16L1BRadiances)
GOES_17_MATCHED = GOES("17", "MATCHED", GOES17L1BRadiances)
GOES_18_MATCHED = GOES("18", "MATCHED", GOES18L1BRadiances)
