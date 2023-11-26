"""
cimr.data.patmosx
=================

This module provides the patmosx input data object, that can be used to extract
daily gridded AVHRR and HIRS observations from the PATMOS-X CDR.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import patmosx_asc, patmosx_des
import xarray as xr

from cimr.data.input import Input, MinMaxNormalized


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

    return xr.Dataset({
        "time": (("time",), time),
        "latitude": (("latitude",), latitude),
        "longitude": (("longitude",), longitude),
        "obs_imager": (dims_imager, obs_imager),
        "obs_sounder": (dims_sounder, obs_sounder),
    })




class PATMOSX(Input, MinMaxNormalized):
    """
    Provides an interface to extract and load training data from the PATMOS-X
    dataset.
    """
    def __init__(self):
        super().__init__("patmosx", 1, ["obs_imager", "obs_sounder"])

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

        if isinstance(domain, dict):
            domain = domain[8]
        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]

        while time < end:
            time_range = TimeRange(
                time,
                time + time_step - timedelta(seconds=1)
            )

            recs_asc = patmosx_asc.find_files(time_range)[:1]
            recs_asc = [rec.get() for rec in recs_asc]
            recs_des = patmosx_des.find_files(time_range)[:1]
            recs_des = [rec.get() for rec in recs_des]

            data_asc = load_observations(recs_asc[0].local_path).rename(
                obs_sounder="obs_sounder_asc",
                obs_imager="obs_imager_asc"
            )
            data_des = load_observations(recs_des[0].local_path).rename(
                obs_sounder="obs_sounder_des",
                obs_imager="obs_imager_des"
            )
            data = xr.merge([data_asc, data_des])[{"time": 0}]
            data = data.interp(latitude=lats, longitude=lons)

            filename = time.strftime("patmosx_%Y%m%d_%H%M.nc")
            data.to_netcdf(output_folder / filename)

            time = time + time_step


patmosx = PATMOSX()
