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
from typing import List, Optional
from zipfile import ZipFile

import numpy as np
from pansat import Product, TimeRange
from pansat.geometry import Geometry
from pansat.products.satellite.meteosat import l1b_msg_seviri, l1b_rs_msg_seviri
from pansat.time import to_datetime64
from satpy import Scene
import xarray as xr

from chimp.areas import Area
from chimp.data import InputDataset
from chimp.data.utils import round_time, get_output_filename


LOGGER = logging.getLogger(__name__)


CHANNEL_CONFIGURATIONS = {
    "ALL": [
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
    ],
    "MATCHED" : [
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


def save_file(dataset, time, time_step, output_folder):
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




class SEVIRIL1B(InputDataset):
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
        super().__init__(name, "seviri", 4, ["obs"])
        self.pansat_product = pansat_product
        if not channel_configuration.upper() in CHANNEL_CONFIGURATIONS:
            raise RuntimeError(
                f"Channel configuration must by one of {list(CHANNEL_CONFIGURATIONS.keys())}."
            )
        self.channel_configuration = channel_configuration.upper()

    @property
    def n_channels(self) -> int:
        return 12

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
        prod = self.pansat_product

        if path is not None:
            all_files = sorted(list(path.glob("**/MSG*")))

        time_range = TimeRange(start_time, end_time)

        if path is not None:
            recs = [
                FileRecord(path, product=prod) for path in all_files
                if prod.matches(path) and prod.get_temporal_coverage(path).covers(time_range)
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

        found_files = list(matched_recs.values())
        return [rec.get().local_path for rec in found_files]

    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given input data file.

        Args:
            path: A Path object pointing to the file to process.
            domain: An area object defining the training domain.
            output_folder: A path pointing to the folder to which to write
                the extracted training data.
            time_step: A timedelta object defining the retrieval time step.
        """
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        if isinstance(domain, Area):
            domain = domain[4]

        with TemporaryDirectory() as tmp:
            with ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(tmp)
            files = list(Path(tmp).glob("*.nat"))
            scene = Scene(files)
            datasets = scene.available_dataset_names()

            start, end = self.pansat_product.get_temporal_coverage(path)
            time_c = start + 0.5 * (end - start)

            datasets = CHANNEL_CONFIGURATIONS[self.channel_configuration]
            scene.load(datasets, generate=False)
            scene_r = scene.resample(domain, radius_of_influence=12e3)

            obs = []
            names = []
            for name in datasets:
                obs.append(scene_r[name].compute().data)

            acq_time = scene[datasets[0]].compute().acq_time.mean().data

            obs = np.stack(obs, -1)
            data = xr.Dataset({"obs": (("y", "x", "channels"), obs)})
            data["time"] = to_datetime64(time_c)
            data["acq_time_mean"] = time_c
            save_file(data, time_c, time_step, output_folder)


SEVIRI = SEVIRIL1B("seviri", l1b_msg_seviri, "all")
SEVIRI_RS = SEVIRIL1B("seviri_rs", l1b_rs_msg_seviri, "all")
