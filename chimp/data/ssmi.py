"""
chimp.data.ssmi
==============

This module provides the ssmi input data object that defines the interface
to extract and load SSMI CDR data.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from pansat import TimeRange
from pansat.geometry import Geometry
from pansat.time import to_datetime64
from pansat.products.satellite.ncei import ssmi_csu_gridded_all
import pyresample
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.utils import get_output_filename
from chimp.data.resample import split_time, resample_and_split


LOGGER = logging.getLogger(__name__)


def load_observations(path: Path) -> xr.Dataset:
    """
    Load SSMI observations from NetCDF file.

    Args:
         path: Path object pointing to a SSMI CDR file.

    Return:
         An xarray.Dataset containing the loaded data.
    """
    channels = [
        "fcdr_tb19h",
        "fcdr_tb19v",
        "fcdr_tb22v",
        "fcdr_tb37h",
        "fcdr_tb37v",
        "fcdr_tb85h",
        "fcdr_tb85v",
    ]

    with xr.open_dataset(path) as data:

        if "fcdr_tb85v_asc" not in data:
            channels = [
                "fcdr_tb19h",
                "fcdr_tb19v",
                "fcdr_tb22v",
                "fcdr_tb37h",
                "fcdr_tb37v",
                "fcdr_tb91h",
                "fcdr_tb91v",
            ]

        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))

        obs_asc = xr.concat(
            [data[chan + "_asc"] for chan in channels], "channels"
        )
        obs_des = xr.concat(
            [data[chan + "_dsc"] for chan in channels], "channels"
        )

        time_asc = data.time_offset_asc.data.astype("timedelta64[s]")
        time_des = data.time_offset_dsc.data.astype("timedelta64[s]")

    data = xr.Dataset({
        "longitude": (("longitude",), data.lon.data),
        "latitude": (("latitude",), data.lat.data),
        "obs_asc": (("latitude", "longitude", "channels"), obs_asc.transpose("lat", "lon", "channels").data),
        "obs_des": (("latitude", "longitude", "channels"), obs_asc.transpose("lat", "lon", "channels").data),
        "second_of_day_asc": (("latitude", "longitude",), time_asc.astype("int64")),
        "second_of_day_des": (("latitude", "longitude"), time_des.astype("int64"))
    })

    return data.transpose("latitude", "longitude", "channels")


class SSMITBS(InputDataset):
    """
    Provides an interface to extract and load training data from the PATMOS-X
    dataset.
    """

    def __init__(self):
        super().__init__(
            "ssmi",
            "ssmi",
            1,
            ["obs"],
            spatial_dims=("latitude", "longitude")
        )
        self.pansat_product = ssmi_csu_gridded_all

    @property
    def n_channels(self) -> int:
        return 14

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
        reference data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP reference data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.grib")))
            matching = [
                path for path in all_files if self.pansat_product.matches(path)
                and prod.get_temporal_coverage(path).covers(TimeRange(time_start, time_end))
            ]
            return matching

        recs = self.pansat_product.get(TimeRange(start_time, end_time))
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
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        try:
            data = load_observations(path)
        except KeyError:
            LOGGER.warning(
                "File %s does not contain the expected variables."
            )
            return None

        start_time, end_time = self.pansat_product.get_temporal_coverage(path)

        if isinstance(domain, Area):
            domain = domain[8]

        lons, lats = domain.get_lonlats()
        regular = (lons[0] == lons[1]).all()
        lons = lons[0]
        lats = lats[..., 0]

        if regular:
            data = data.interp(latitude=lats, longitude=lons)
            data["time_asc"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["second_of_day_asc"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data_asc = split_time(data, "time_asc", start_time, end_time, np.timedelta64(6, "h"))
            data["time_des"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["second_of_day_des"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data_des = split_time(data, "time_des", start_time, end_time, np.timedelta64(6, "h"))
        else:
            data["time"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["second_of_day_asc"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data_asc = resample_and_split(data, domain, time_step, radius_of_influence=10e3)
            data_asc = data.drop_vars(["latitude", "longitude"])
            data_asc = data.transpose("y", "x", "time", "channels")

            data["time"] = (
                ("latitude", "longitude"),
                (
                    to_datetime64(start_time) +
                    data["second_of_day_des"].data.astype("int64").astype("timedelta64[s]")
                )
            )
            data_des = resample_and_split(data, domain, time_step, radius_of_influence=10e3)
            data_des = data.drop_vars(["latitude", "longitude"])
            data_des = data.transpose("y", "x", "time", "channels")

        filename = get_output_filename(
            "patmosx", start_time, time_step
        )

        output_file = output_folder / filename
        if output_file.exists():
            output_data = xr.load_dataset(output_file)

            mask = np.isfinite(data_asc["obs_asc"].data)
            output_data["obs"].data[mask] = data_asc["obs_asc"].data[mask]
            mask = np.isfinite(data_des["obs_des"].data)
            output_data["obs"].data[mask] = data_des["obs_des"].data[mask]

            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in output_data.variables if obs != "time_of_day"
            }
            output_data.to_netcdf(output_file, encoding=encodings)
        else:
            mask = np.isfinite(data_des["obs_des"].data)
            data_asc["obs_asc"].data[mask] = data_des["obs_asc"].data[mask]
            data_asc = data_asc.drop_vars(["obs_des"]).rename(obs_asc="obs")
            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in data_asc.variables if obs != "time_of_day"
            }
            data_asc.to_netcdf(output_file, encoding=encodings)


SSMI = SSMITBS()
