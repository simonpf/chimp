"""
chimp.data.gridsat
=================

This module provides the GridSat class that provides an interface to extract
training data from the GridSat-B1 dataset.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import gridsat_b1
import xarray as xr

from chimp.data.input import Input


def load_gridsat_data(path):
    """
    Load GridSat observations.

    Loads GridSat visible, IR water vapor, and IR window observations and
    combines them into a single xarray.Dataset with dimensions time,
    latitude, longitude and channels.

    Args:
         path: Path object pointing to the GridSat B1 file to load.

    Return:
         An xarray.Dataset containing the loaded data.
    """
    with xr.open_dataset(path) as data:
        time = data["time"].data
        lons = data["lon"].data
        lats = data["lat"].data
        irwin = data["irwin_cdr"].data
        irwvp = data["irwvp"].data
        vschn = data["vschn"].data
    return xr.Dataset({
        "longitude": (("longitude",), lons),
        "latitude": (("latitude",), lats),
        "time": (("time",), time),
        "obs": (
            ("time", "latitude", "longitude", "channels"),
            np.stack([vschn, irwvp, irwin], -1)
        )
    })



class GridSat(Input):
    """
    Provides an interface to extract and load training data from the GridSat
    B1 dataset.
    """
    def __init__(self):
        super().__init__("gridsat", 1, "obs")
        self.n_channels = 24

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
        output_folder = Path(output_folder) / "gridsat"
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
            if time_step.total_seconds() > 3 * 60 * 60:
                time_range = TimeRange(
                    time,
                    time + time_step - timedelta(hours=1, minutes=30, seconds=1)
                )
            else:
                time_range = TimeRange(
                    time,
                    time + time_step - timedelta(seconds=1)
                )

            recs = gridsat_b1.find_files(time_range)
            recs = [rec.get() for rec in recs]
            gridsat_data = xr.concat(
                [load_gridsat_data(rec.local_path) for rec in recs],
                dim="time"
            )
            gridsat_data = gridsat_data.interp(
                latitude=lats,
                longitude=lons
            )
            if time_step.total_seconds() < 3 * 60 * 60:
                gridsat_data = gridsat_data.interp(time=time)

            filename = time.strftime("gridsat_%Y%m%d_%H_%M.nc")

            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in gridsat_data.variables
            }
            gridsat_data.to_netcdf(output_folder / filename, encoding=encodings)

            time = time + time_step


gridsat = GridSat()
