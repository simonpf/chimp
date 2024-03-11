"""
chimp.data.gpm
=============

This module implements functions the handling of data from the
GPM constellation.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, List, Optional

import numpy as np
import pansat
from pansat import Geometry
from pansat.granule import merge_granules
from pansat.time import TimeRange, to_datetime64, to_timedelta64
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
    l2b_gpm_cmb
)
from pyresample import AreaDefinition
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.utils import get_output_filename
from chimp.utils import round_time


LOGGER = logging.getLogger(__name__)


class GPML1CData(InputDataset):
    """
    Represents  input data derived from GPM L1C products.
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
        InputDataset.__init__(self, name, name, scale, "tbs", n_dim=2)
        self.products = products
        self.n_swaths = n_swaths
        self.n_channels = n_channels
        self.radius_of_influence = radius_of_influence


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
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [prod.matches(path.filename) for path in all_files]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.get(start_time, end_time)
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

        input_data = self.products[0].open(path)
        swath_data =  []

        for swath in range(1, self.n_swaths + 1):
            new_names = {
                f"latitude_s{swath}": "latitude",
                f"longitude_s{swath}": "longitude",
                f"tbs_s{swath}": "tbs",
                f"scan_time": "time",
                f"channels_s{swath}": "channels",
            }
            data = input_data.rename(new_names)[
                ["latitude", "longitude", "tbs", "time"]
            ]
            if f"pixels_s{swath}" in data:
                data = data.rename({f"pixels_s{swath}": "pixels"})

            # Need to expand scan time to full dimensions.
            time, _ = xr.broadcast(data.time, data.longitude)
            data["time"] = time

            swath_data.append(
                resample_and_split(
                    data,
                    domain[self.scale],
                    time_step,
                    self.radius_of_influence,
                    include_swath_center_coords=True
                )
            )

        data = xr.concat(swath_data, "channels")
        for time_ind  in range(data.time.size):

            data_t = data[{"time": time_ind}]

            tbs = data_t.tbs.data
            if np.isfinite(tbs).any(-1).sum() < 100:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    time
                )
                continue

            comp = {
                "dtype": "uint16",
                "scale_factor": 0.01,
                "zlib": True,
                "_FillValue": 2**16 - 1,
            }
            encoding = {
                "tbs": comp,
                "col_inds_swath_center": {"dtype": "int16"},
                "row_inds_swath_center": {"dtype": "int16"},
            }
            filename = get_output_filename(
                self.name, data.time[time_ind].data, time_step
            )

            LOGGER.info(
                "Writing training samples to %s.",
                output_folder / filename
            )
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


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


#
# GPM CMB
#


class GPMCMB(ReferenceDataset):
    """
    Represents retrieval reference data derived from GPM combined radar-radiometer retrievals.
    """
    def __init__(self):
        self.name = "cmb"
        super().__init__(
            self.name,
            4,
            RetrievalTarget("surface_precip"),
            quality_index=None
        )
        self.products = [l2b_gpm_cmb]
        self.scale = 4
        self.radius_of_influence = 6e3

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
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [prod.matches(path.filename) for path in all_files]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.get(start_time, end_time)
        return [rec.local_path for rec in recs]

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

        input_data = self.products[0].open(path)
        data = input_data.rename({
            "scan_time": "time",
            "estim_surf_precip_tot_rate": "surface_precip"
        })

        # Need to expand scan time to full dimensions.
        time, _ = xr.broadcast(data.time, data.longitude)
        data["time"] = time

        data = resample_and_split(
            data,
            domain[self.scale],
            time_step,
            radius_of_influence=self.radius_of_influence,
            include_swath_center_coords=True
        )

        for time_ind  in range(data.time.size):

            data_t = data[{"time": time_ind}]

            precip = data_t.surface_precip.data
            if np.isfinite(precip).any(-1).sum() < 100:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    time
                )
                continue

            encoding = {
                "surface_precip": {"dtype": "float32", "zlib": True},
                "col_inds_swath_center": {"dtype": "int16"},
                "row_inds_swath_center": {"dtype": "int16"},
            }
            filename = get_output_filename(
                self.name, data.time[time_ind].data, time_step
            )

            LOGGER.info(
                "Writing training samples to %s.",
                output_folder / filename
            )
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


CMB = GPMCMB()
