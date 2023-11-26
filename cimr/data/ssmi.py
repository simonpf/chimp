"""
cimr.data.ssmi
==============

This module provides the ssmi input data object that defines the interface
to extract and load SSMI CDR data.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import ssmi_csu
import pyresample
from pyresample.geometry import SwathDefinition
from pyresample import kd_tree
import xarray as xr

from cimr.data.input import Input, MinMaxNormalized


def load_observations(path: Path) -> xr.Dataset:
    """
    Load SSMI observations from NetCDF file.

    Args:
         path: Path object pointing to a SSMI CDR file.

    Return:
         An xarray.Dataset containing the loaded data.
    """
    with xr.open_dataset(path) as data:

        scan_time = data["scan_time_lores"].data
        latitude_lores = data["lat_lores"].data
        longitude_lores = data["lon_lores"].data
        latitude_hires = data["lat_hires"].data
        longitude_hires = data["lon_hires"].data

        obs_lores = np.stack(
            [
                data["fcdr_tb19h"].data,
                data["fcdr_tb19v"].data,
                data["fcdr_tb22v"].data,
                data["fcdr_tb37h"].data,
                data["fcdr_tb37v"].data,
            ],
            axis=-1
        )
        obs_hires = np.stack(
            [
                data["fcdr_tb85h"].data,
                data["fcdr_tb85v"].data,
            ],
            axis=-1
        )

    return xr.Dataset({
        "scan_time": (("scans_lores",), scan_time),
        "latitude_lores": (("scans_lores", "pixels_lores"), latitude_lores),
        "longitude_lores": (("scans_lores", "pixels_lores"), longitude_lores),
        "obs_lores": (("scans_lores", "pixels_lores", "channels_lores"), obs_lores),
        "latitude_hires": (("scans_hires", "pixels_hires"), latitude_hires),
        "longitude_hires": (("scans_hires", "pixels_hires"), longitude_hires),
        "obs_hires": (("scans_hires", "pixels_hires", "channels_hires"), obs_hires),
    })



def resample_data(
        dataset: xr.Dataset,
        target_grid: pyresample.AreaDefinition,
        output_data: Optional[xr.Dataset],
        orbit_part="asc"
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
        output_data = xr.Dataset({
            "latitude": (("latitude",), lats_t[:, 0]),
            "longitude": (("longitude",), lons_t[0]),
            "obs_lores": (("latitude", "longitude", "channels_lores"), obs_lores),
            "obs_hires": (("latitude", "longitude", "channels_hires"), obs_hires),
        })

    for res in ["lores", "hires"]:

        lons = dataset[f"longitude_{res}"].data
        lats = dataset[f"latitude_{res}"].data

        dlat = np.zeros(lats.shape[:1])
        dlat[1:] = np.diff(lats.mean(-1))
        dlat[0] = dlat[1]
        orbit = dlat > 0
        if orbit_part == "des":
            orbit = ~orbit

        swath = SwathDefinition(lons=lons[orbit], lats=lats[orbit])
        target = SwathDefinition(
            lons=lons_t,
            lats=lats_t
        )

        info = kd_tree.get_neighbour_info(
                swath,
                target,
                radius_of_influence=64e3,
                neighbours=1
        )
        ind_in, ind_out, inds, _ = info

        obs =  dataset[f"obs_{res}"].data[orbit]

        obs_r = kd_tree.get_sample_from_neighbour_info(
            'nn',
            target.shape,
            obs,
            ind_in,
            ind_out,
            inds,
            fill_value=np.nan
        )

        valid = np.isfinite(obs_r)
        output_data[f"obs_{res}"].data[valid] = obs_r[valid]

    return output_data





class SSMI(Input, MinMaxNormalized):
    """
    Provides an interface to extract and load training data from the PATMOS-X
    dataset.
    """
    def __init__(self):
        super().__init__("ssmi", 1, ["obs_asc", "obs_des"])

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
        output_folder = Path(output_folder) / "ssmi"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        time = datetime(year=year, month=month, day=day)
        end = time + timedelta(days=1)

        if isinstance(domain, dict):
            domain = domain[16]


        while time < end:
            time_range = TimeRange(
                time,
                time + time_step - timedelta(seconds=1)
            )

            recs = ssmi_csu.find_files(time_range)
            recs = [rec.get() for rec in recs]

            data_asc = None
            data_des = None
            for rec in recs:
                print(rec.local_path)
                data = load_observations(rec.local_path)
                data_asc = resample_data(
                    data,
                    domain,
                    data_asc,
                    "asc"
                )
                data_des = resample_data(
                    data,
                    domain,
                    data_des,
                    "des"
                )

            data_asc = data_asc.rename(
                obs_lores="obs_lores_asc",
                obs_hires="obs_hires_asc",
            )
            data_des = data_des.rename(
                obs_lores="obs_lores_des",
                obs_hires="obs_hires_des",
            )

            data = xr.merge(
                [data_asc, data_des],
            )

            filename = time.strftime("ssmi_%Y%m%d_%H%M.nc")
            data.to_netcdf(output_folder / filename)

            time = time + time_step


ssmi = SSMI()
