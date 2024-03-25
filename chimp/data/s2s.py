"""
chimp.data.s2s
==============

This modules implements a reference data class to extract baseline
results from ECMWF S2S database.
"""
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional


import numpy as np
import xarray as xr
import pansat
from pansat import TimeRange
from pansat.geometry import Geometry
from pansat.products.model.ecmwf import (
    s2s_ecmwf_total_precip,
    s2s_ecmwf_total_precip_10,
    s2s_ukmo_total_precip,
    s2s_ukmo_total_precip_3,
)

from chimp.areas import Area
from chimp.data.utils import get_output_filename, round_time
from chimp.data.reference import ReferenceDataset


class S2SForecast(ReferenceDataset):
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
        reference data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP reference data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.grib")))
            matching = []
            for prod in [self.control_product, self.ensemble_product]:
                matching += [
                        path for path in all_files if prod.matches(path)
                        and prod.get_temporal_coverage(path).covers(TimeRange(time_start, time_end))
                ]
            return matching

        recs = []
        for prod in [self.control_product, self.ensemble_product]:
            recs += prod.get(TimeRange(start_time, end_time))
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

        prod = [prod for prod in [self.control_product, self.ensemble_product] if prod.matches(path)][0]
        data = prod.open(path)

        start_time = round_time(data.time.data.min(), time_step)
        end_time = round_time(data.time.data.max(), time_step)
        time = start_time

        print("TIMES", start_time, end_time)

        while time < end_time:

            if not time in data.time:
                time += time_step
                continue

            lons, lats = domain[8].get_lonlats()
            regular = (lons[0] == lons[1]).all()
            lons = lons[0]
            lats = lats[..., 0]

            ind = np.where(data.time == time)[0][0]
            data_t = data[{"time": ind}]

            data_t = data_t.interp(longitude=lons, latitude=lats)

            if np.timedelta64(0, "h") not in data_t.step:
                steps = np.concatenate([[np.timedelta64(0, "h")], data_t.step.data])
                data_t = data_t.interp(step=steps, method="nearest", kwargs={"fill_value": 0.0})

            ensemble = False
            if "number" in data_t.dims:
                data_t = data_t.mean("number")
                ensemble = True

            precip = np.diff(data_t.tp.data, axis=0)

            output_filename = get_output_filename(self.name, time, time_step)
            output_file = output_folder / output_filename
            varname = "precipitation" if not ensemble else "precipitation_em"

            if output_file.exists():
                output_data = xr.load_dataset(output_file)
                output_data[varname] = (("step", "latitude", "longitude"), precip)
            else:
                output_data = xr.Dataset({
                    "latitude": (("latitude",), data_t.latitude.data),
                    "longitude": (("longitude",), data_t.longitude.data),
                    "step": (("step",), data_t.step.data[:-1]),
                    varname: (("step", "latitude", "longitude"), precip)
                })

            output_data.to_netcdf(
                output_folder / output_filename,
            )
            time += time_step


ECMWF = S2SForecast("s2s_ecmwf", s2s_ecmwf_total_precip, s2s_ecmwf_total_precip_10)
UKMO = S2SForecast("s2s_ukmo", s2s_ukmo_total_precip, s2s_ukmo_total_precip_3)
