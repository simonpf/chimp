"""
chimp.data.daily_precip
=======================

This module implements a reference data class providing global, daily
 precipitation accumulations derived from PERISANN and IMERG data.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from pansat.download.providers import IowaStateProvider
from pansat.products.satellite import persiann, gpm

from pansat.time import TimeRange

from pansat.time import to_datetime64, to_datetime
from pansat.geometry import Geometry
import xarray as xr

from chimp.areas import Area
from chimp.utils import round_time
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.utils import get_output_filename
from chimp import areas


LOGGER = logging.getLogger(__name__)


precip = RetrievalTarget("precipitation", None)


SPLIT = datetime(2000, 6, 1)


class DailyPrecip(ReferenceDataset):
    """
    The DailyPrecip dataset consists a climatology of daily, quasi-global
    precipitation accumulations. The accumulations are derived from the
    PERSIANN CDR dataset from 1983 until June 2000 and from IMERG following
    that.
    """
    def __init__(
            self,
            scale: int,
    ):
        """
        Args:
            scale: The native scale of the precipitation dataset.
        """
        super().__init__("daily_precip", scale, [precip], None)


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
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*")))
            matching = []
            matching += [
                path for path in all_files
                if persiann.cdr_daily.matches(path)
                and persiann.cdr_daily.get_temporal_coverage(path).start < SPLIT - timedelta(hours=1)
                and persiann.cdr_daily.get_temporal_coverage(path).covers(TimeRange(start_time, end_time))
            ]
            matching += [
                path for path in all_files
                if gpm.l3b_day_3imerg_ms_mrg_v07.matches(path)
                and gpm.l3b_day_3imerg_ms_mrg_v07.get_temporal_coverage(path).start >= SPLIT
                and gpm.l3b_day_3imerg_ms_mrg_v07.get_temporal_coverage(path).covers(TimeRange(start_time, end_time))

            ]
            return matching

        matching = []

        start_time = to_datetime(start_time)
        end_Time = to_datetime(end_time)

        print("LSTART :: ", min(end_time, SPLIT))
        print("RSTART :: ", max(start_time, SPLIT))
        if start_time < SPLIT:
            matching += [
                rec.get().local_path for rec in persiann.cdr_daily.get(
                    TimeRange(start_time, min(end_time, SPLIT - timedelta(hours=1)))
                )
            ]
        if end_time > SPLIT:
            matching += [
                rec.get().local_path for rec in gpm.l3b_day_3imerg_ms_mrg_v07.get(
                    TimeRange(max(start_time, SPLIT), end_time)
                )
            ]
        return matching

    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given source file.

        Args:
           path: A Path object pointing to the file to process.
           domain: An area object defining the training domain.
           output_folder: A path pointing to the folder to which to write
               the extracted training data.
           time_step: A timedelta object defining the retrieval time step.
        """
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        if persiann.cdr_daily.matches(path):
            data = persiann.cdr_daily.open(path)
            time = persiann.cdr_daily.get_temporal_coverage(path).start
        else:
            data = gpm.l3b_day_3imerg_ms_mrg_v07.open(path)
            time = gpm.l3b_day_3imerg_ms_mrg_v07.get_temporal_coverage(path).start
            data["surface_precipitation"].attrs = {}
            data["latitude"].attrs = {}
            data["longitude"].attrs = {}
            data["time"].attrs = {}
            data = data.rename(surface_precipitation="precipitation")

        if isinstance(domain, Area):
            domain = domain[self.scale]

        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[:, 0]
        data = data.interp(latitude=lats, longitude=lons)[{"time": 0}]

        output_filename = get_output_filename(
            "precip", time, time_step
        )
        encoding = {
            "latitude": {"dtype": "float32"},
            "longitude": {"dtype": "float32"},
            "precipitation": {"dtype": "float32"},
        }
        data.to_netcdf(
            output_folder / output_filename,
            encoding=encoding
        )


DAILY_PRECIP = DailyPrecip(1)
