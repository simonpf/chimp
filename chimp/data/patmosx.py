"""
chimp.data.patmosx
=================

This module provides the patmosx input data object, that can be used to extract
daily gridded AVHRR and HIRS observations from the PATMOS-X CDR.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import patmosx_asc, patmosx_des
import xarray as xr

from chimp.data.utils import get_output_filename
from chimp.data.input import InputDataset


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
            ["obs_imager", "obs_sounder"],
            spatial_dims=("latitude", "longitude")
        )

    @property
    def n_channels(self) -> int:
        return 92

    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(days=1),
            include_scan_time=False
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
        if time_step.total_seconds() != 24 * 60 * 60:
            raise ValueError(
                "PATMOS-X observations can only be extracted at time steps "
                " of one day."
            )
        output_folder = Path(output_folder) / "patmosx"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        time = datetime(year=year, month=month, day=day)
        end = time + timedelta(days=1)

        lons, lats = domain[8].get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]

        while time < end:
            time_range = TimeRange(
                time,
                time + time_step - timedelta(seconds=1)
            )

            recs = patmosx_asc.find_files(time_range)
            recs += patmosx_des.find_files(time_range)
            recs = [rec.get() for rec in recs]

            if len(recs) == 0:
                LOGGER.warning(
                    "Didn't find any Patmos-X observations for %s.",
                    time
                )
                time = time + time_step
                continue

            obs_sounder = np.nan * np.zeros((lats.size, lons.size, 4, 8), np.float32)
            obs_imager = np.nan * np.zeros((lats.size, lons.size, 4, 15), np.float32)
            time_bins = np.arange(0, 25, 6) * 3600


            for rec in recs:
                data = load_observations(rec.local_path)[{"time": 0}]
                data = data.interp(latitude=lats, longitude=lons)
                data = data.transpose("latitude", "longitude", ...)
                scan_time = data.scan_line_time.data

                for ind in range(4):
                    lower = time_bins[ind]
                    upper = time_bins[ind + 1]
                    mask = (scan_time >= lower) * (scan_time < upper)

                    data_sounder = data["obs_sounder"].data
                    valid = mask * np.isfinite(data_sounder).any(-1)
                    obs_sounder[valid, ind] = data["obs_sounder"].data[valid]

                    data_imager = data["obs_imager"].data
                    valid = mask * np.isfinite(data_imager).any(-1)
                    obs_imager[valid, ind] = data["obs_imager"].data[valid]


            time_of_day = time_bins[:-1] + 0.5 * (time_bins[1:] - time_bins[:-1])

            results = xr.Dataset({
                "latitude": (("latitude",), lats),
                "longitude": (("longitude",), lons),
                "time_of_day": (("time_of_day"), time_of_day.astype("timedelta64[s]")),
                "obs_sounder": (("latitude", "longitude", "time_of_day", "channels_sounder"), obs_sounder),
                "obs_imager": (("latitude", "longitude", "time_of_day", "channels_imager"), obs_imager)
            })
            output_filename = get_output_filename(
                "patmosx", time, time_step.total_seconds() // 60
            )
            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in results.variables if obs != "time_of_day"
            }
            results.to_netcdf(output_folder / output_filename, encoding=encodings)

            time = time + time_step


patmosx = PATMOSX()
