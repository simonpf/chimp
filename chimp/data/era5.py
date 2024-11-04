"""
chimp.data.era5
===============

This module provides functionality to extract ERA5 precipitation as reference data.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
from pansat import Geometry
from pansat.time import to_datetime, to_datetime64
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.utils import get_output_filename, records_to_paths
from chimp.data.reference import ReferenceDataset, RetrievalTarget


FILENAME_REGEXP = re.compile(
    r"ERA5_(\d\d\d\d)(\d\d)(\d\d)_surf.nc"
)


def get_start_and_end_time(filename: str) -> Tuple[datetime, datetime]:
    """
    Determine start and end time of ERA5 file.

    Args:
        filename: The name of the ERA5 file.

    Return:
        A tuple cotaining start and end-time of the file.

    """
    match = FILENAME_REGEXP.match(filename)
    if match is None:
        raise ValueError(
            f"Provided filename '{filename}' doesn't match ERA5 file format.",
        )

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(hours=23, minutes=59, seconds=59)
    return (start_time, end_time)


class ERA5Data(ReferenceDataset):
    """
    Input data from ERA5.
    """
    def __init__(
            self,
    ):
        super().__init__("era5_surface_precip", 4, [RetrievalTarget("surface_precip")], quality_index=None)
        self.dense = True


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
        start_time = to_datetime(start_time)
        end_time = to_datetime(end_time)

        if path is None:
            raise ValueError(
                "Path to local ERA5 data is required."
            )

        path = Path(path)
        files = sorted(list(path.glob("**/*_surf.nc")))
        matches = []

        for path in files:
            match = FILENAME_REGEXP.match(path.name)
            if match is not None:
                start_time_obs, end_time_obs = get_start_and_end_time(path.name)
                if (start_time <= end_time_obs) and (start_time_obs <= end_time):
                    matches.append(path)

        return matches


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
        path = records_to_paths(path)

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        with xr.open_dataset(path) as data:
            data = data[["tp"]]

        data.tp.data *= 1e3

        data = data.rename(tp="surface_precip")
        lons = data.longitude.data
        lons[lons > 180] -= 360
        data = data.sortby("longitude")

        data_s = resample_and_split(
            data,
            domain[self.scale],
            time_step,
            30e3,
        )

        if data_s is None:
            return None

        for time_ind  in range(data_s.time.size):

            data_t = data_s[{"time": time_ind}]

            comp = {
                "dtype": "uint16",
                "scale_factor": 0.001,
                "zlib": True,
                "_FillValue": 2**16 - 1,
            }
            encoding = {"surface_precip": comp,}

            filename = get_output_filename(
                self.name, data_t.time.data, time_step
            )
            output_file = output_folder / filename
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


ERA5 = ERA5Data()
