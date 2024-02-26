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
import pansat
from pansat import TimeRange
from pansat.download.providers import ecmwf
from pansat.products.model.ecmwf import (
    s2s_ecmwf_total_precip,
    s2s_ecmwf_total_precip_10,
    s2s_ukmo_total_precip,
    s2s_ukmo_total_precip_3,
)

from chimp.data.utils import get_output_filename
from chimp.data.reference import ReferenceData


class S2SForecast(ReferenceData):
    """
    Interface class to extract baseline forecasts from the ECMWF S2S database.
    """
    def __init__(
        self,
        name: str,
        control_product: pansat.Product,
        ensemble_product: pansat.Product
    ):
        """
        Args:
            scale: The native scale of the precipitation dataset.
        """
        self.scale = 32
        self.control_product = control_product
        self.ensemble_product = ensemble_product
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
            recs = self.control_product.find_files(time_range)
            if len(recs) > 0:

                rec = recs[0].get()
                data = self.control_product.open(rec)

                time_64 = np.datetime64(time.strftime("%Y-%m-%dT%H:%M:%S"))
                if time_64 in data.time:

                    ind = np.where(data.time == time_64)[0][0]
                    data_t = data[{"time": ind}]

                    if np.timedelta64(0, "h") not in data_t.step:
                        steps = np.concatenate([[np.timedelta64(0, "h")], data_t.step.data])
                        data_t = data_t.interp(step=steps, method="nearest", kwargs={"fill_value": 0.0})
                    precip = np.diff(data_t.tp.data, axis=0)

                    recs_ens = self.ensemble_product.get(time_range)
                    data_ens = self.ensemble_product.open(recs_ens[0])
                    data_ens = data_ens[{"time": ind}]
                    if np.timedelta64(0, "h") not in data_ens.step:
                        steps = np.concatenate([[np.timedelta64(0, "h")], data_ens.step.data])
                        data_ens = data_ens.interp(step=steps, method="nearest", kwargs={"fill_value": "extrapolate"})
                    data_ens = data_ens.mean("number")
                    precip_ens = np.diff(data_t.tp.data, axis=0)

                    data_t = xr.Dataset({
                        "latitude": (("latitude",), data_t.latitude.data),
                        "longitude": (("longitude",), data_t.longitude.data),
                        "step": (("step",), data_t.step.data[:-1]),
                        "precipitation": (("step", "latitude", "longitude"), precip),
                        "precipitation_em": (("step", "latitude", "longitude"), precip_ens)
                    })
                    data_t = data_t.resample(step="1D").sum()
                    data_t = data_t.interp(
                        longitude=lons,
                        latitude=lats
                    )

                    output_filename = get_output_filename(
                        self.name, time, time_step.total_seconds() // 60
                    )

                    data_t.to_netcdf(
                        output_folder / output_filename,
                    )

            time = time + time_step


s2s_ecwmf = S2SForecast("s2s_ecmwf", s2s_ecmwf_total_precip, s2s_ecmwf_total_precip_10)
s2s_ukmo = S2SForecast("s2s_ukmo", s2s_ukmo_total_precip, s2s_ukmo_total_precip_3)
