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
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from pansat.download.providers import IowaStateProvider
from pansat.products.satellite import persiann, gpm

from pansat.time import TimeRange

from pansat.time import to_datetime64
import xarray as xr

from chimp.utils import round_time
from chimp.data.reference import ReferenceData, RetrievalTarget
from chimp.data.utils import get_output_filename
from chimp import areas


LOGGER = logging.getLogger(__name__)


precip = RetrievalTarget("precipitation", None)


class DailyPrecip(ReferenceData):
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


    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(minutes=15),
            include_scan_time=False
    ):
        """
        Extract daily precipitation accumulations.

        Args:
            year: The year
            month: The month
            day: The day
            output_folder: The folder to which to write the extracted
                observations.
            path: Not used, included for compatibility.
            time_step: Time step defining the temporal resolution at which to extract
                training samples.
            include_scan_time: Ignored. Included for compatibility.
        """
        output_folder = Path(output_folder) / "daily_precip"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=24)
        time = start_time
        files = []
        while time < end_time:
            if time < datetime(2020, 6, 1):
                recs = persiann.cdr_daily.find_files(TimeRange(time, time))
                if len(recs) == 0:
                    LOGGER.warning(
                        "Didn't find any PERSIANN files for %s.",
                        time
                    )
                    time = time + time_step
                    continue

                rec = recs[0].get()
                data = persiann.cdr_daily.open(rec)
            else:
                recs = gpm.l3b_day_3imerg_ms_mrg_v07.find_files(TimeRange(time, time))
                if len(recs) == 0:
                    LOGGER.warning(
                        "Didn't find any IMERG files for %s.",
                        time
                    )
                    time = time + time_step
                    continue

                rec = recs[0].get()
                data = gpm.l3b_day_3imerg_ms_mrg_v06.find_files(TimeRange(time, time))
                data = data.rename({
                    "lon": "longitude",
                    "lat": "latitude",
                    "precipitationCal": "precipitation"
                })["precipitation"]


            if isinstance(domain, dict):
                domain = domain[self.scale]

            lons, lats = domain.get_lonlats()
            lons = lons[0]
            lats = lats[:, 0]
            data = data.interp(latitude=lats, longitude=lons)[{"time": 0}]

            output_filename = get_output_filename(
                "precip", time, time_step.total_seconds() // 60
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

            time = time + time_step


daily_precip = DailyPrecip(1)
