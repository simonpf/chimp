"""
chimp.data.ssmi
==============

This module provides the ssmi input data object that defines the interface
to extract and load SSMI CDR data.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import ssmi_csu_gridded_all
import pyresample
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.utils import get_output_filename


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


class SSMI(InputDataset):
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

    @property
    def n_channels(self) -> int:
        return 14

    def process_day(
        self,
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(days=1),
        include_scan_time=False,
    ):
        """
        Extract training data for a given day.

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
        output_folder = Path(output_folder) / "ssmi"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        time = datetime(year=year, month=month, day=day)
        end = time + timedelta(days=1)

        if isinstance(domain, Area):
            domain = domain[16]

        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[:, 0]
        time_bins = (np.arange(0, 25, 6) * 3600)

        while time < end:
            time_range = TimeRange(time, time + time_step - timedelta(seconds=1))

            prod = ssmi_csu_gridded_all

            recs = prod.find_files(time_range)
            tbs = np.nan * np.zeros((lats.size, lons.size, 4, 7), np.float32)

            for rec in recs:
                rec = rec.get()
                data = load_observations(rec.local_path)
                data = data.interp(
                    latitude=lats,
                    longitude=lons,
                    method="nearest"
                )

                for ind in range(4):
                    lower = time_bins[ind]
                    upper = time_bins[ind + 1]

                    sod = data.second_of_day_asc.data
                    mask = (lower <= sod) * (sod < upper)
                    valid = mask * np.isfinite(data.obs_asc.data).any(-1)
                    tbs[valid, ind] = data.obs_asc.data[valid]

                    sod = data.second_of_day_des.data
                    mask = (lower <= sod) * (sod < upper)
                    valid = mask * np.isfinite(data.obs_des.data).any(-1)
                    tbs[valid, ind] = data.obs_des.data[valid]



            if len(recs) > 0:
                time_of_day = 0.5 * (time_bins[1:] + time_bins[:-1])
                training_sample = xr.Dataset({
                    "latitude": (("latitude",), lats),
                    "longitude": (("longitude",), lons),
                    "time_of_day": (("time_of_day"), time_of_day.astype(
                        "timedelta64[s]"
                    )),
                    "obs": (("latitude", "longitude", "time_of_day", "channels"), tbs)
                })
                output_filename = get_output_filename(
                    "ssmi", time, minutes=1440
                )
                encodings = {
                    obs: {"dtype": "float32", "zlib": True}
                    for obs in training_sample.variables
                }
                training_sample.to_netcdf(
                    output_folder / output_filename,
                    encoding=encodings
                )

            time = time + time_step


ssmi = SSMI()
