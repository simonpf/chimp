"""
chimp.data.gpm
=============

This module implements functions the handling of data from the
GPM constellation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, List, Optional

import numpy as np
from quantnn.normalizer import Normalizer
import pansat
from pansat.catalog import Index
from pansat.granule import merge_granules
from pansat.time import TimeRange
from pansat.roi import find_overpasses
from pansat.products.satellite.gpm import (
    l1c_gpm_gmi,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_npp_atms,
    l1c_noaa20_atms,
    l1c_f16_ssmis,
    l1c_f17_ssmis,
    l1c_f18_ssmis,
    l1c_gcomw1_amsr2,
)
from pyresample import AreaDefinition

from chimp.data.input import Input, MinMaxNormalized
from chimp.data.resample import resample_tbs
from chimp.utils import round_time

import xarray as xr


def save_gpm_scene(
    name: str,
    time: np.datetime64,
    scene: xr.Dataset,
    output_folder: Path,
    time_step: np.timedelta64,
):
    """
    Save training data scene.

    Args:
        name: The name of the GPM sensor.
        time: The (mean) time of the overpass.
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        scene: A xarray.Dataset containing the resampled observations.
        output_folder: The folder to which to write the training data.
        time_step: A time difference defining the time step to which times
            are rounded.
    """
    minutes = time_step.seconds // 60
    time_15 = round_time(time, minutes=minutes)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"{name}_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    scene = scene.transpose("y", "x", "channels", "swath_centers")

    comp = {
        "dtype": "uint16",
        "scale_factor": 0.01,
        "zlib": True,
        "_FillValue": 2**16 - 1,
    }
    encoding = {
        "tbs": comp,
        "swath_center_col_inds": {"dtype": "int16"},
        "swath_center_row_inds": {"dtype": "int16"},
    }

    if output_filename.exists():
        data = xr.load_dataset(output_filename)
        mask = np.any(np.isfinite(scene.tbs.data), -1)
        data.tbs.data[mask] = scene.tbs.data[mask]
        data = data.transpose("y", "x", "channels", "swath_centers")
        data.to_netcdf(output_filename, encoding=encoding)
    else:
        scene.to_netcdf(output_filename, encoding=encoding)
    return None


def process_overpass(
    name: str,
    domain: AreaDefinition,
    scene: xr.Dataset,
    n_swaths: int,
    radius_of_influence: float,
    output_folder: Path,
    time_step: timedelta,
    include_scan_time: bool = False,
    min_valid: int = 100,
) -> None:
    """
    Resample TBs in overpass to domain and save data.

    Args:
        name: The name of the sensor.
        domain: A domain dict describing the area for which to extract
            the training data.
        scene: An 'xarray.Dataset' containing a single overpass over
            the domain.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
        time_step: The time step used for the discretization of the input
            data.
        include_scan_time: Boolean flag indicating whether or not to include
            the resampled scan time in the extracted retrieval input.
    """
    tbs_r = resample_tbs(
        domain,
        scene,
        radius_of_influence=radius_of_influence,
        include_scan_time=include_scan_time,
        n_swaths=n_swaths,
    )
    time = scene.scan_time.mean().data.item()
    tbs_r.attrs = scene.attrs

    n_valid = np.any(np.isfinite(tbs_r.tbs.data), -1).sum()
    if n_valid < min_valid:
        return None
    save_gpm_scene(name, time, tbs_r, output_folder, time_step)


class GPML1CData(Input, MinMaxNormalized):
    """
    Represents all input data derived from GPM L1C products.
    """

    def __init__(
        self,
        name: str,
        scale: int,
        products: List[pansat.Product],
        n_swaths: int,
        n_channels: int,
        radius_of_influence,
    ):
        """
        Args:
            name: Name of the sensor.
            scale: The spatial scale to which the output will be mapped.
            products: The pansat products reprsenting the L1C products of the
                sensors.
            n_swaths: The number of swaths in the L1C files.
            n_channels: The number of channels in the input data.
            radius_of_influence: The radius of influence in meters to use for the
                resampling of the data.
        """
        MinMaxNormalized.__init__(self, name)
        Input.__init__(self, name, scale, "tbs", n_dim=2)
        self.products = products
        self.n_swaths = n_swaths
        self.n_channels = n_channels
        self.radius_of_influence = radius_of_influence

    def process_day(
        self,
        domain: dict,
        year: int,
        month: int,
        day: int,
        output_folder: Path,
        path: Path = Optional[None],
        time_step: timedelta = timedelta(minutes=15),
        include_scan_time=False,
    ):
        """
        Extract GMI input observations for the CHIMP retrieval.

        Args:
            domain: A domain dict specifying the area for which to
                extract GMI input data.
            year: The year.
            month: The month.
            day: The day.
            output_folder: The root of the directory tree to which to write
                the training data.
            path: Not used, included for compatibility.
            time_step: The time step between consecutive retrieval steps.
            include_scan_time: If set to 'True', the resampled scan time will
                be included in the extracted training input.
        """
        output_folder = Path(output_folder) / self.name
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
        time_range = TimeRange(start_time, end_time)

        for product in self.products:
            product_files = product.find_files(time_range, roi=domain["roi_poly"])

            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                for rec in product_files:
                    print(rec.filename)
                    rec = rec.download(tmp)
                    index = Index.index(product, [rec.local_path])
                    granules = index.find(roi=domain["roi_poly"])
                    granules = merge_granules(granules)

                    for granule in granules:
                        scene = granule.open()
                        process_overpass(
                            self.name,
                            domain[self.scale],
                            scene,
                            self.n_swaths,
                            self.radius_of_influence,
                            output_folder,
                            time_step,
                            include_scan_time,
                            min_valid=1_000 / np.sqrt(self.scale),
                        )


GMI = GPML1CData("gmi", 4, [l1c_gpm_gmi], 2, 13, 15e3)
ATMS = GPML1CData("atms", 16, [l1c_noaa20_atms, l1c_npp_atms], 4, 9, 64e3)

MHS_PRODUCTS = [
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
]
MHS = GPML1CData("mhs", 8, MHS_PRODUCTS, None, 5, 64e3)

SSMIS_PRODUCTS = [
    l1c_f16_ssmis,
    l1c_f17_ssmis,
    l1c_f17_ssmis,
]
SSMIS = GPML1CData("ssmis", 8, SSMIS_PRODUCTS, 4, 11, 30e3)

AMSR2 = GPML1CData("amsr2", 4, [l1c_gcomw1_amsr2], 6, 12, 30e3)
