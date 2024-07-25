"""
chimp.data.imerg
================

This module defines IMERG reference data that can be used as baseline
results to evaluate retrievals.
"""
from pathlib import Path
from typing import List, Optional


import h5py
import numpy as np
import pansat
from pansat import TimeRange
from pansat.geometry import Geometry
from pansat.products.satellite.gpm import (
    l3b_imerg_half_hourly_early,
    l3b_imerg_half_hourly_late,
    l3b_imerg_half_hourly_final,
)
import xarray as xr


from chimp.areas import Area
from chimp.data.utils import round_time
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.resample import resample_and_split
from chimp.data.utils import records_to_paths, get_output_filename


class IMERG(ReferenceDataset):
    """
    Represents retrieval reference data derived from GPM IMERG data.
    """
    def __init__(self, variant: str, pansat_product: pansat.Product):
        name = f"imerg_{variant.lower()}"
        super().__init__(name, 4, [RetrievalTarget("surface_precip")])
        self.product = pansat_product


    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
        """
        Find IMERG reference data files within a given file range.

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
        recs = []
        times = np.arange(
            round_time(start_time, time_step),
            round_time(end_time, time_step) + np.timedelta64(1, "s"),
            time_step
        )
        for time in times:
            recs += self.product.find_files(TimeRange(time))

        return recs


    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract reference data samples from a given IMERG  file.

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

        input_data = self.product.open(path)
        input_data = input_data.rename(surface_precipitation="surface_precip")
        input_data.time.attrs = {}
        input_data.longitude.attrs = {}
        input_data.latitude.attrs = {}
        input_data.surface_precip.attrs = {}

        data = resample_and_split(
            input_data,
            domain[self.scale],
            time_step,
            20e3,
        )

        for time_ind in range(data.time.size):

            data_t = data[{"time": time_ind}]

            comp = {
                "dtype": "uint16",
                "scale_factor": 0.01,
                "zlib": True,
                "_FillValue": 2**16 - 1,
            }
            filename = get_output_filename(self.name, data_t.time.data, time_step)

            encoding = {
                "surface_precip": comp
            }
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


IMERG_EARLY = IMERG(variant="early", pansat_product=l3b_imerg_half_hourly_early)
IMERG_LATE = IMERG(variant="late", pansat_product=l3b_imerg_half_hourly_late)
IMERG_FINAL = IMERG(variant="final", pansat_product=l3b_imerg_half_hourly_final)
