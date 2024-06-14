"""
chimp.data.opera
================

Defines the OPERA input data classes for loading precipitation and
 reflectivity estimates from OPERA composites.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr

from pansat import TimeRange, Product
from pansat.geometry import Geometry
from pansat.products.ground_based.opera import reflectivity, surface_precip

from chimp.areas import Area
from chimp.data import ReferenceDataset
from chimp.data.reference import RetrievalTarget
from chimp.data.resample import resample_data
from chimp.data.utils import round_time, get_output_filename


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
            quality_index="qi"
        )

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
        prod = self.pansat_product
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.nc")))
            matching = [
                    path for path in all_files if prod.matches(path)
                    and prod.get_temporal_coverage(path).covers(TimeRange(time_start, time_end))
            ]
            return matching

        recs = prod.get(TimeRange(start_time, end_time))
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

        prod = self.pansat_product
        data = prod.open(path)
        start_time, end_time = prod.get_temporal_coverage(path)
        start_time = round_time(start_time, time_step)
        end_time = round_time(end_time, time_step)
        time = start_time
        while (time < end_time):

            data_t = data.interp(time=time, method="nearest")
            data_interp = data.interp(time=time)

            if "reflectivity" in data_interp:
                invalid = data_interp.reflectivity.data < -9e6
                data_interp.reflectivity.data[invalid] = np.nan
                noise = data_interp.reflectivity.data < -30
                data_interp.reflectivity.data[noise] = -30

            data_r = resample_data(data_interp, domain[4], radius_of_influence=4e3)
            data_r = data_r.rename({"quality_indicator": "qi"})

            output_filename = get_output_filename(
                self.target_name, time, time_step
            )

            encoding = {
                "reflectivity": {
                    "dtype": "uint16",
                    "add_offset": 0,
                    "scale_factor": 0.01,
                    "_FillValue": -1,
                    "zlib": True,
                },
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
            encoding = {name: encoding[name] for name in data_r.variables}
            data_r.to_netcdf(output_folder / output_filename, encoding=encoding)

            time = time + time_step


OPERA_REFLECTIVITY = Opera("reflectivity", reflectivity)
OPERA_SURFACE_PRECIP = Opera("surface_precip", surface_precip)
