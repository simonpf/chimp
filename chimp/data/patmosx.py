"""
chimp.data.patmosx
=================

This module provides the patmosx input data object, that can be used to extract
daily gridded AVHRR and HIRS observations from the PATMOS-X CDR.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
from pansat import TimeRange
from pansat.geometry import Geometry
from pansat.time import to_datetime64
from pansat.products.satellite.ncei import patmosx_asc, patmosx_des
import xarray as xr

from chimp.areas import Area
from chimp.data.utils import get_output_filename
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split, split_time


LOGGER = logging.getLogger()


def load_observations(path):
    """
    Load AVHHR and HIRS data from PATMOS-X file.

    Args:
         path: Path object pointing to PATMOS-X file.

    Return:
         An xarray.Dataset containing the loaded data.
    """
    with xr.open_dataset(path) as data:

        time = data["time"].data
        latitude = data["latitude"].data
        longitude = data["longitude"].data

        obs_imager = np.stack(
            [
                data["refl_0_65um_nom"].data,
                data["refl_0_86um_nom"].data,
                data["refl_1_60um_nom"].data,
                data["refl_3_75um_nom"].data,
                data["temp_4_46um_nom"].data,
                data["temp_4_52um_nom"].data,
                data["temp_6_7um_nom"].data,
                data["temp_7_3um_nom"].data,
                data["temp_9_7um_nom"].data,
                data["temp_11_0um_nom"].data,
                data["temp_12_0um_nom"].data,
                data["temp_13_3um_nom"].data,
                data["temp_13_6um_nom"].data,
                data["temp_13_9um_nom"].data,
                data["temp_14_2um_nom"].data
            ],
            axis=-1
        )

        obs_sounder = np.stack(
            [
                data["temp_3_75um_nom_sounder"].data,
                data["temp_4_45um_nom_sounder"].data,
                data["temp_4_57um_nom_sounder"].data,
                data["temp_11_0um_nom_sounder"].data,
                data["temp_12_0um_nom_sounder"].data,
                data["temp_14_5um_nom_sounder"].data,
                data["temp_14_7um_nom_sounder"].data,
                data["temp_14_9um_nom_sounder"].data,
            ],
            axis=-1
        )

    dims_imager = ("time", "latitude", "longitude", "channels_imager")
    dims_sounder = ("time", "latitude", "longitude", "channels_sounder")

    scan_line_time = data["scan_line_time"].dt.seconds.astype("float32").data

    return xr.Dataset({
        "time": (("time",), time),
        "latitude": (("latitude",), latitude),
        "longitude": (("longitude",), longitude),
        "scan_line_time": (("time", "latitude", "longitude"), scan_line_time),
        "obs_imager": (dims_imager, obs_imager),
        "obs_sounder": (dims_sounder, obs_sounder),
    })




class PATMOSX(InputDataset):
    """
    Provides an interface to extract and load training data from the PATMOS-X
    dataset.
    """
    def __init__(self):
        super().__init__(
            "patmosx",
            "patmosx",
            1,
            ["obs_imager_asc", "obs_imager_des", "obs_sounder_asc", "obs_sounder_des"],
            spatial_dims=("latitude", "longitude")
        )
        self.pansat_products = [patmosx_asc, patmosx_des]

    @property
    def n_channels(self) -> int:
        return 46

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
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
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.nc")))
            matching = []
            for prod in self.pansat_products:
                matching += [
                        path for path in all_files if prod.matches(path)
                        and prod.get_temporal_coverage(path).covers(TimeRange(time_start, time_end))
                ]
            return matching

        recs = []
        for prod in self.pansat_products:
            recs += prod.get(TimeRange(start_time, end_time))
        return [rec.local_path for rec in recs]

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
        prod = [prod for prod in self.pansat_products if prod.matches(path)][0]

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        lons, lats = domain[8].get_lonlats()
        regular = (lons[0] == lons[1]).all()
        lons = lons[0]
        lats = lats[..., 0]

        data = load_observations(path)[{"time": 0}]

        start_time, end_time = prod.get_temporal_coverage(path)

        if regular:
            data = data.interp(latitude=lats, longitude=lons)
            data["time"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["scan_line_time"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data = split_time(data, "time", start_time, end_time, np.timedelta64(6, "h"))
        else:
            data["time"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["scan_line_time"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data = resample_and_split(data, domain, time_step, radius_of_influence=10e3)
            data = data.drop_vars(["latitude", "longitude"])
            data = data.transpose("y", "x", "time", "channels_sounder", "channels_imager")
            data["time"].data[:] = time

        filename = get_output_filename(
            "patmosx", start_time, time_step
        )

        output_file = output_folder / filename
        if output_file.exists():
            output_data = xr.load_dataset(output_file)
            for var in ["obs_imager", "obs_sounder"]:
                mask = np.isfinite(data[var].data)
                output_data[var].data[mask] = data[var].data[mask]
            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in data.variables if obs != "time_of_day"
            }
            output_data.to_netcdf(output_file, encoding=encodings)
        else:
            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in data.variables if obs != "time_of_day"
            }
            data.to_netcdf(output_file, encoding=encodings)

PATMOSX = PATMOSX()
