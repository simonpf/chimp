"""
chimp.data.cloudsat
===================

This module provides access to CloudSat reference estimates.
"""

from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
from pansat import TimeRange, Geometry
from pansat.time import to_datetime, to_datetime64
from pansat.products.satellite.cloudsat import l2c_rain_profile, l2c_snow_profile
import xarray as xr

from chimp.areas import Area
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.resample import resample_and_split
from chimp.data.utils import get_output_filename, records_to_paths


class CloudSatSurfacePrecip(ReferenceDataset):
    """
    Surface precipitation estimates from cloudsat.
    """
    def __init__(
            self,
    ):
        super().__init__("cloudsat_surface_precip", 4, [RetrievalTarget("surface_precip")], quality_index=None)


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
        time_range = TimeRange(start_time, end_time)
        recs = l2c_rain_profile.find_files(time_range, roi=roi.roi)
        recs += l2c_snow_profile.find_files(time_range, roi=roi.roi)
        return recs


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

        if l2c_snow_profile.matches(path):
            cs_data = l2c_snow_profile.open(path)[["time", "latitude", "longitude", "surface_precip"]]
        else:
            cs_data = l2c_rain_profile.open(path)[["time", "latitude", "longitude", "surface_precip"]]
        cs_data = cs_data.drop_vars(("rays", "surface_elevation"))
        cs_data["surface_precip"] = cs_data.surface_precip.astype(np.float32)
        sp = cs_data.surface_precip.data
        sp[sp < 0] = np.nan

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        data_s = resample_and_split(
            cs_data,
            domain[self.scale],
            time_step,
            2e3,
        )

        if data_s is None:
            return None

        for time_ind  in range(data_s.time.size):

            data_t = data_s[{"time": time_ind}]

            if np.isfinite(data_t.surface_precip.data).sum() < 1:
                continue

            comp = {
                "dtype": "float32",
                "zlib": True,
            }
            encoding = {
                "surface_precip": comp,
            }

            filename = get_output_filename(
                self.name, data_t.time.data, time_step
            )
            output_file = output_folder / filename

            if output_file.exists():
                output_data = xr.load_dataset(output_file)
                mask_new = np.isfinite(data_t.surface_precip.data)
                mask_existing = np.isfinite(output_data.surface_precip.data)
                mask_both = mask_new * mask_existing
                output_data["surface_precip"].data[mask_both] += data_t.surface_precip.data[mask_both]
                mask_new_only = mask_new * ~mask_existing
                output_data["surface_precip"].data[mask_new_only] = data_t.surface_precip.data[mask_new_only]

                mask = np.isfinite(output_data["surface_precip"].data)
                if mask.sum() > 0:
                    row_inds_swath_center, col_inds_swath_center = np.where(mask)
                    output_data = output_data.drop_vars(
                        ("row_inds_swath_center", "col_inds_swath_center")
                    )
                    output_data["row_inds_swath_center"] = (("center_indices",), row_inds_swath_center)
                    output_data["col_inds_swath_center"] = (("center_indices",), col_inds_swath_center)

                output_data.to_netcdf(output_file, encoding=encoding)
            else:
                mask = np.isfinite(data_t["surface_precip"].data)
                if mask.sum() > 0:
                    row_inds_swath_center, col_inds_swath_center = np.where(mask)
                    data_t["row_inds_swath_center"] = (("center_indices",), row_inds_swath_center)
                    data_t["col_inds_swath_center"] = (("center_indices",), col_inds_swath_center)
                data_t.to_netcdf(output_file, encoding=encoding)


    def find_random_scene(
            self,
            path,
            rng,
            multiple=4,
            scene_size=256,
            valid_fraction=0.2,
            **kwargs
    ):
        """
        Finds a random crop in the CloudSat data that is guaranteeed to have
        valid observations.

        Args:
            path: The path of the reference data file.
            rng: A numpy random generator instance to randomize the scene search.
            multiple: Limits the scene position to coordinates that are multiples
                of the given value.
            valid_fraction: The minimum amount of valid samples in the
                region.

        Return:
            A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
            of the random crop.
        """
        try:
            with xr.open_dataset(path) as data:
                if "latitude" in data.dims:
                    n_rows = data.latitude.size
                    n_cols = data.longitude.size
                else:
                    n_rows = data.y.size
                    n_cols = data.x.size

                row_inds = data.row_inds_swath_center.data
                col_inds = data.col_inds_swath_center.data

                valid = (
                    (row_inds > scene_size // 2)
                    * (row_inds < n_rows - scene_size // 2)
                    * (col_inds > scene_size // 2)
                    * (col_inds < n_cols - scene_size // 2)
                )

                if valid.sum() == 0:
                    return None

                row_inds = row_inds[valid]
                col_inds = col_inds[valid]

                ind = rng.choice(np.arange(valid.sum()))
                row_c = row_inds[ind]
                col_c = col_inds[ind]

                i_start = int((row_c - scene_size // 2) // multiple * multiple)
                i_end = int(i_start + scene_size)
                j_start = int((col_c - scene_size // 2) // multiple * multiple)
                j_end = int(j_start + scene_size)

                if rng is not None:
                    offset = rng.integers(
                        -min(i_start, scene_size // 2),
                        min(n_rows - i_end, scene_size // 2)
                    )
                    i_start += offset
                    i_end += offset

                    offset = rng.integers(
                        -min(j_start, scene_size // 2),
                        min(n_cols - j_end, scene_size // 2)
                    )
                    j_start += offset
                    j_end += offset

            return (i_start, i_end, j_start, j_end)
        except OSError:
            return None

CLOUDSAT_SURFACE_PRECIP = CloudSatSurfacePrecip()


class CloudSatSnow(ReferenceDataset):
    """
    Surface precipitation estimates from cloudsat.
    """
    def __init__(
            self,
    ):
        super().__init__("cloudsat_snow", 4, [RetrievalTarget("snow")], quality_index=None)


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
        time_range = TimeRange(start_time, end_time)
        recs = l2c_snow_profile.find_files(time_range, roi=roi.roi)
        return recs


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

        cs_data = l2c_snow_profile.open(path)[["time", "latitude", "longitude", "surface_precip"]]
        cs_data = cs_data.drop_vars(("rays", "surface_elevation"))
        cs_data["surface_precip"] = cs_data.surface_precip.astype(np.float32)
        cs_data = cs_data.rename(surface_precip="snow")
        sp = cs_data.snow.data
        sp[sp < 0] = np.nan

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        data_s = resample_and_split(
            cs_data,
            domain[self.scale],
            time_step,
            2e3,
        )

        if data_s is None:
            return None

        for time_ind  in range(data_s.time.size):

            data_t = data_s[{"time": time_ind}]

            if np.isfinite(data_t.snow.data).sum() < 1:
                continue

            comp = {
                "dtype": "float32",
                "zlib": True,
            }
            encoding = {
                "snow": comp,
            }

            filename = get_output_filename(
                self.name, data_t.time.data, time_step
            )
            output_file = output_folder / filename

            if output_file.exists():
                output_data = xr.load_dataset(output_file)
                mask_new = np.isfinite(data_t.snow.data)
                mask_existing = np.isfinite(output_data.snow.data)
                mask_both = mask_new * mask_existing
                output_data["snow"].data[mask_both] += data_t.snow.data[mask_both]
                mask_new_only = mask_new * ~mask_existing
                output_data["snow"].data[mask_new_only] = data_t.snow.data[mask_new_only]

                mask = np.isfinite(output_data["snow"].data)
                if mask.sum() > 0:
                    row_inds_swath_center, col_inds_swath_center = np.where(mask)
                    output_data = output_data.drop_vars(
                        ("row_inds_swath_center", "col_inds_swath_center")
                    )
                    output_data["row_inds_swath_center"] = (("center_indices",), row_inds_swath_center)
                    output_data["col_inds_swath_center"] = (("center_indices",), col_inds_swath_center)

                output_data.to_netcdf(output_file, encoding=encoding)
            else:
                mask = np.isfinite(data_t["snow"].data)
                if mask.sum() > 0:
                    row_inds_swath_center, col_inds_swath_center = np.where(mask)
                    data_t["row_inds_swath_center"] = (("center_indices",), row_inds_swath_center)
                    data_t["col_inds_swath_center"] = (("center_indices",), col_inds_swath_center)
                data_t.to_netcdf(output_file, encoding=encoding)


    def find_random_scene(
            self,
            path,
            rng,
            multiple=4,
            scene_size=256,
            valid_fraction=0.2,
            **kwargs
    ):
        """
        Finds a random crop in the CloudSat data that is guaranteeed to have
        valid observations.

        Args:
            path: The path of the reference data file.
            rng: A numpy random generator instance to randomize the scene search.
            multiple: Limits the scene position to coordinates that are multiples
                of the given value.
            valid_fraction: The minimum amount of valid samples in the
                region.

        Return:
            A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
            of the random crop.
        """
        try:
            with xr.open_dataset(path) as data:
                if "latitude" in data.dims:
                    n_rows = data.latitude.size
                    n_cols = data.longitude.size
                else:
                    n_rows = data.y.size
                    n_cols = data.x.size

                row_inds = data.row_inds_swath_center.data
                col_inds = data.col_inds_swath_center.data

                valid = (
                    (row_inds > scene_size // 2)
                    * (row_inds < n_rows - scene_size // 2)
                    * (col_inds > scene_size // 2)
                    * (col_inds < n_cols - scene_size // 2)
                )

                if valid.sum() == 0:
                    return None

                row_inds = row_inds[valid]
                col_inds = col_inds[valid]

                ind = rng.choice(np.arange(valid.sum()))
                row_c = row_inds[ind]
                col_c = col_inds[ind]

                i_start = int((row_c - scene_size // 2) // multiple * multiple)
                i_end = int(i_start + scene_size)
                j_start = int((col_c - scene_size // 2) // multiple * multiple)
                j_end = int(j_start + scene_size)

                if rng is not None:
                    offset = rng.integers(
                        -min(i_start, scene_size // 2),
                        min(n_rows - i_end, scene_size // 2)
                    )
                    i_start += offset
                    i_end += offset

                    offset = rng.integers(
                        -min(j_start, scene_size // 2),
                        min(n_cols - j_end, scene_size // 2)
                    )
                    j_start += offset
                    j_end += offset

            return (i_start, i_end, j_start, j_end)
        except OSError:
            return None


CLOUDSAT_SNOW = CloudSatSnow()
