"""
chimp.data.seviri
================

Functionality for reading and processing SEVIRI data.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pansat import Product, TimeRange
from pansat.roi import any_inside
from pansat.download.providers.eumetsat import EUMETSATProvider
from pansat.products.satellite.meteosat import l1b_msg_seviri, l1b_rs_msg_seviri
from pansat.time import to_datetime64
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr

from chimp.data import Input
from chimp.data.utils import get_output_filename


LOGGER = logging.getLogger(__name__)


CHANNEL_CONFIGURATIONS = {
    "all": [
        "HRV",
        "VIS006",
        "VIS008",
        "IR_016",
        "IR_039",
        "WV_062",
        "WV_073",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
    ]
}


def save_file(dataset, time_step, output_folder):
    """
    Save a training data file.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            SEVIRI observations.
        time_step: A timedelta object specifying the temporal sampling
            of the training data.
        output_folder: The folder to which to write the training data.
    """
    filename = get_output_filename("seviri", dataset["time"].data, time_step)
    output_filename = Path(output_folder) / filename
    comp = {"dtype": "int16", "scale_factor": 0.05, "zlib": True, "_FillValue": -99}
    encoding = {"obs": comp}
    dataset.to_netcdf(output_filename, encoding=encoding)


def download_and_resample_data(
    product: Product,
    time: datetime,
    domain: geometry.AreaDefinition,
    channel_configuration: str,
) -> xr.Dataset:
    """
    Download, load and resample SEVIRI data.

    Args:
        product: A pansat.Product defining the SEVIRI product to download.
        time: A datetime object specifying the time for which to download
            SEVIRI data.
        domain: An AreaDefinition object specifying the domain over which to
            extract the satellite observations.
        channel_configuration: A string specifying the SEVIRI channel
            configuration.

    Return:
        An xarray.Dataset containing the loaded SEVIRI observations.
    """
    time_range = TimeRange(time - timedelta(minutes=2), time + timedelta(minutes=2))
    recs = product.get(time_range)
    closest = time_range.find_closest_ind([rec.temporal_coverage for rec in recs])
    rec = recs[closest[0]].get()

    with TemporaryDirectory() as tmp:
        with ZipFile(rec.local_path, "r") as zip_ref:
            zip_ref.extractall(tmp)
        files = list(Path(tmp).glob("*.nat"))
        scene = Scene(files)
        datasets = scene.available_dataset_names()

        datasets = CHANNEL_CONFIGURATIONS[channel_configuration]
        scene.load(datasets)
        scene_r = scene.resample(domain, radius_of_influence=4e3)

        obs = []
        names = []
        for name in datasets:
            obs.append(scene_r[name].compute().data)

        acq_time = scene[datasets[0]].compute().acq_time.mean().data

        obs = np.stack(obs, -1)

        data = xr.Dataset({"obs": (("y", "x", "channels"), obs)})
        data["time"] = to_datetime64(time)
        data["acq_time_mean"] = acq_time
        return data


class SEVIRIInputData(Input):
    """
    Input class implementing an interface to extract and load SEVIRI input
    data.
    """

    def __init__(self, name: str, pansat_product: Product, channel_configuration: str):
        """
        Args:
            name: The name of the input
            pansat_product: The pansat.Product representing the data product
                from which to extract the observations.
            channel_configuration: A 'str' specifying the channel configuration.
        """
        super().__init__(name, 4, ["obs"])
        self.pansat_product = pansat_product
        self.channel_configuration = channel_configuration

    @property
    def n_channels(self) -> int:
        return 12

    def process_day(
        self,
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(minutes=15),
        include_scan_time=False,
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
        output_folder = Path(output_folder) / "seviri"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        existing_files = [
            f.name for f in output_folder.glob(f"goes_{year}{month:02}{day:02}*.nc")
        ]

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
        time = start_time
        while time < end_time:
            output_filename = get_output_filename("seviri", time, time_step)
            if not (output_folder / output_filename).exists():
                dataset = download_and_resample_data(
                    self.pansat_product,
                    time,
                    domain[self.scale],
                    self.channel_configuration,
                )
                save_file(dataset, time_step, output_folder)

            time = time + time_step


seviri_rs = SEVIRIInputData("seviri_rs", l1b_rs_msg_seviri, "all")
seviri = SEVIRIInputData("seviri", l1b_msg_seviri, "all")
