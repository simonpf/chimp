"""
chimp.data.s2s
==============

This modules implements a reference data class to extract baseline
results from ECMWF S2S database.
"""
from pathlib import Path
from datetime import datetime, timedelta


import numpy as np
import xarray as xr
from pansat import TimeRange
from pansat.download.providers import ecmwf
from pansat.products.model.ecmwf import (
    s2s_ecmwf_total_precip
)

from chimp.data.utils import get_output_filename
from chimp.data.reference import ReferenceData

class S2SForecast(ReferenceData):
    """
    Interface class to extract baseline forecasts from the ECMWF S2S database.
    """
    def __init__(
        self,
        name,
    ):
        """
        Args:
            scale: The native scale of the precipitation dataset.
        """
        self.scale = 32
        super().__init__(name, self.scale, None, None)


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
        Extract s2s forecasts.

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
        output_folder = Path(output_folder) / self.name
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=24)

        if isinstance(domain, dict):
            domain = domain[self.scale]

        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[:, 0]

        time = start_time
        while time < end_time:
            time_range = TimeRange(time, time + timedelta(days=1))
            recs = s2s_ecmwf_total_precip.find_files(time_range)
            if len(recs) > 0:
                rec = recs[0].get()
                data = xr.load_dataset(rec.local_path)

                time_64 = np.datetime64(time.strftime("%Y-%m-%dT%H:%M:%S"))
                if time_64 in data.time:
                    ind = np.where(data.time == time_64)[0][0]
                    data_t = data[{"time": ind}].resample(step="1d").sum()
                    data_t = data_t.interp(
                        longitude=lons,
                        latitude=lats
                    )
                    data_t = data_t.rename({
                        "tp": "precipitation"
                    })

                    output_filename = get_output_filename(
                        self.name, time, time_step.total_seconds() // 60
                    )

                    data_t.to_netcdf(
                        output_folder / output_filename,
                    )

            time = time + time_step


s2s_ecwmf = S2SForecast("s2s_ecmwf")
