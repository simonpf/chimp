"""
chimp.data.opera
================

Defines the OPERA input data classes for loading precipitation and
 reflectivity estimates from OPERA composites.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Optional

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr

from pansat import TimeRange, Product
from pansat.products.ground_based.opera import reflectivity, surface_precip

from chimp.areas import Area
from chimp.data import ReferenceDataset
from chimp.data.reference import RetrievalTarget
from chimp.utils import round_time
from chimp.data.resample import resample_data
from chimp.data.utils import get_output_filename


class Opera(ReferenceDataset):
    """
    Represents reference data derived from OPERA radar composites.
    """

    def __init__(
            self,
            target_name: str,
            pansat_product: Product
    ):
        self.target_name = target_name
        self.pansat_product = pansat_product
        super().__init__(
            "opera_" + self.target_name,
            scale=4,
            targets=[RetrievalTarget(target_name)],
            quality_index="quality_indicator"
        )

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
        Extract OPERA reference data observations for the CHIMP retrieval.

        Args:
            domain: A domain dict specifying the area for which to
                extract OPERA data.
            year: The year.
            month: The month.
            day: The day.
            output_folder: The root of the directory tree to which to write
                the training data.
            path: Not used, included for compatibility.
            time_step: The time step between consecutive retrieval steps.
            include_scan_time: Not used; included for compatibility.
        """
        output_folder = Path(output_folder) / self.name
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day, second=1)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=58)
        time_range = TimeRange(start_time, end_time)

        time = start_time

        if isinstance(domain, Area):
            domain = domain[self.scale]

        file_recs = self.pansat_product.get(time_range)
        if len(file_recs) == 0:
            raise RuntimeError(f"Didn't find any OPERA files for {year}-{month}-{day}.")
        if len(file_recs) > 1:
            raise RuntimeError(
                f"Found more that one OPERA file for {year}-{month}-{day}."
            )

        opera_data = self.pansat_product.open(file_recs[0])

        while time < end_time:

            opera_data_interp = opera_data.interp(time=time)

            if "reflectivity" in opera_data_interp:
                invalid = opera_data_interp.reflectivity.data < -9e6
                opera_data_interp.reflectivity.data[invalid] = np.nan
                noise = opera_data_interp.reflectivity.data < -30
                opera_data_interp.reflectivity.data[noise] = -30

            data_r = resample_data(opera_data_interp, domain, radius_of_influence=4e3)
            data_r = data_r.drop_vars(["latitude", "longitude"])
            data_r = data_r.rename({"quality_indicator": "qi"})

            output_filename = get_output_filename(
                self.target_name, time, minutes=time_step.total_seconds() // 60
            )
            encoding = {
                "reflectivity": {
                    "dtype": "uint8",
                    "add_offset": -30,
                    "scale_factor": 0.3,
                    "_FillValue": 255,
                    "zlib": True,
                },
                "qi": {
                    "add_offset": 0.0,
                    "scale_factor": 1.0 / 200.0,
                    "zlib": True,
                    "dtype": "uint8",
                    "_FillValue": 255,
                },
            }
            data_r.to_netcdf(output_folder / output_filename, encoding=encoding)
            time = time + time_step


opera = Opera("reflectivity", reflectivity)
opera = Opera("surface_precip", surface_precip)
