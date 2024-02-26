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
from pansat.products.satellite.ncei import ssmi_csu_gridded, ssmis_csu_gridded
import pyresample
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr

from chimp.data.input import Input
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

        obs_lores_asc = xr.concat(
            [data[chan + "_asc"] for chan in channels[:5]], "channels_lores"
        )
        obs_lores_des = xr.concat(
            [data[chan + "_dsc"] for chan in channels[:5]], "channels_lores"
        )
        obs_hires_asc = xr.concat(
            [data[chan + "_asc"] for chan in channels[5:]], "channels_hires"
        )
        obs_hires_des = xr.concat(
            [data[chan + "_dsc"] for chan in channels[5:]], "channels_hires"
        )

    data = xr.Dataset({
        "longitude": (("longitude",), data.lon.data),
        "latitude": (("latitude",), data.lat.data),
        "obs_lores_asc": (("latitude", "longitude", "channels_lores"), obs_lores_asc.transpose("lat", "lon", "channels_lores").data),
        "obs_lores_des": (("latitude", "longitude", "channels_lores"), obs_lores_des.transpose("lat", "lon", "channels_lores").data),
        "obs_hires_asc": (("latitude", "longitude", "channels_hires"), obs_hires_asc.transpose("lat", "lon", "channels_hires").data),
        "obs_hires_des": (("latitude", "longitude", "channels_hires"), obs_hires_des.transpose("lat", "lon", "channels_hires").data)
    })

    return data.transpose("latitude", "longitude", "channels_lores", "channels_hires")


def resample_data(
    dataset: xr.Dataset,
    target_grid: pyresample.AreaDefinition,
    output_data: Optional[xr.Dataset],
    orbit_part="asc",
):
    """
    Resample xarray.Dataset data to global grid.

    Args:
        dataset: xr.Dataset containing data to resample to global grid.
        target_grid: A pyresample.AreaDefinition defining the global grid
            to which to resample the data.

    Return:
        An xarray.Dataset containing the give dataset resampled to
        the global grid.
    """
    lons_t, lats_t = target_grid.get_lonlats()

    if output_data is None:
        obs_lores = np.nan * np.zeros(lons_t.shape + (5,))
        obs_hires = np.nan * np.zeros(lons_t.shape + (2,))
        output_data = xr.Dataset(
            {
                "latitude": (("latitude",), lats_t[:, 0]),
                "longitude": (("longitude",), lons_t[0]),
                "obs_lores": (("latitude", "longitude", "channels_lores"), obs_lores),
                "obs_hires": (("latitude", "longitude", "channels_hires"), obs_hires),
            }
        )

    for res in ["lores", "hires"]:
        lons = dataset[f"longitude_{res}"].data
        lats = dataset[f"latitude_{res}"].data

        dlat = np.zeros(lats.shape[:1])
        dlat[1:] = np.diff(lats.mean(-1))
        dlat[0] = dlat[1]
        orbit = dlat > 0
        if orbit_part == "des":
            orbit = ~orbit

        if orbit.sum() == 0:
            continue

        swath = SwathDefinition(lons=lons[orbit], lats=lats[orbit])
        target = SwathDefinition(lons=lons_t, lats=lats_t)

        info = kd_tree.get_neighbour_info(
            swath, target, radius_of_influence=64e3, neighbours=1
        )
        ind_in, ind_out, inds, _ = info

        obs = dataset[f"obs_{res}"].data[orbit]

        obs_r = kd_tree.get_sample_from_neighbour_info(
            "nn", target.shape, obs, ind_in, ind_out, inds, fill_value=np.nan
        )

        valid = np.isfinite(obs_r)
        output_data[f"obs_{res}"].data[valid] = obs_r[valid]

    return output_data


class SSMI(Input):
    """
    Provides an interface to extract and load training data from the PATMOS-X
    dataset.
    """

    def __init__(self):
        super().__init__(
            "ssmi",
            1,
            ["obs_lores_asc", "obs_lores_des", "obs_hires_asc", "obs_hires_des"],
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

        if isinstance(domain, dict):
            domain = domain[16]

        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[:, 0]

        while time < end:
            time_range = TimeRange(time, time + time_step - timedelta(seconds=1))

            if time > datetime(2015, 1, 1):
                prod = ssmis_csu_gridded
            else:
                prod = ssmi_csu_gridded

            recs = prod.find_files(time_range)

            if len(recs) > 0:
                rec = recs[0].get()
                data = load_observations(rec.local_path)
                data = data.interp(
                    latitude=lats,
                    longitude=lons
                )
                filename = time.strftime("ssmi_%Y%m%d_%H%M.nc")

                output_filename = get_output_filename(
                    "ssmi", time, minutes=1440
                )
                encodings = {
                    obs: {"dtype": "float32", "zlib": True}
                    for obs in data.variables
                }
                data.to_netcdf(output_folder / output_filename, encoding=encodings)

            time = time + time_step


ssmi = SSMI()
