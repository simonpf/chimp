"""
chimp.data.avhrr
================

This module provides access to AVHRR observations.
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


FILENAME_REGEXP = re.compile(
    r"clavrx_\w*\.\w*\.\w*\.D(\d\d)(\d\d\d)\.S(\d\d)(\d\d)\.E(\d\d)(\d\d)\.\w*\.\w*\.[\w_]*\.\w*\.nc"
)


def get_start_and_end_time(filename: str) -> Tuple[datetime, datetime]:
    """
    Determine start and end time of AVHRR file.

    Args:
        filename: The name of the AVHRR file.

    Return:
        A tuple cotaining start and end-time of the file.

    """
    match = FILENAME_REGEXP.match(filename)
    if match is None:
        raise ValueError(
            f"Provided filename '{filename}' doesn't match AVHRR format.",
        )

    year = int(match.group(1)) + 2000
    jday = int(match.group(2))
    start_hour = int(match.group(3))
    start_minute = int(match.group(4))
    end_hour = int(match.group(4))
    end_minute = int(match.group(5))
    start_time = datetime(year, 1, 1) + timedelta(days=jday - 1, hours=start_hour, minutes=start_minute)
    end_time = datetime(year, 1, 1) + timedelta(days=jday - 1, hours=end_hour, minutes=end_minute)
    return (start_time, end_time)


class AVHRRData(InputDataset):
    """
    Input data from AVHRR.
    """
    def __init__(
            self,
    ):
        self.scene_offset = 10
        super().__init__("avhrr", "avhrr", 4, "observations", n_dim=2)
        self.n_channels = 2


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
                "Path to local AVHRR data is required."
            )

        path = Path(path)
        files = path.glob("**/clavrx*.nc")
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

        start_time_obs, end_time_obs = get_start_and_end_time(path.name)
        start_time_obs = to_datetime64(start_time_obs).astype("datetime64[ns]")
        vars = [
            "latitude",
            "longitude",
            "temp_11_0um_nom",
            "temp_12_0um_nom",
            "scan_line_time"
        ]
        data = xr.load_dataset(path)[vars].rename(
            scan_lines_along_track_direction="scans",
            pixel_elements_along_scan_direction="pixels"
        )


        start_time_obs = start_time_obs.astype("datetime64[D]").astype("datetime64[ns]")
        time = start_time_obs + data["scan_line_time"]
        time.data[time.data < time.data[0]] += np.timedelta64(1, "D")

        print("PR :: ", path, time.data[0], time.data[-1])
        time, _ = xr.broadcast(time, data.latitude)
        data["time"] = (("scans", "pixels"), time.data)

        temp_11 = data["temp_11_0um_nom"]
        temp_12 = data["temp_12_0um_nom"]
        data = data.drop_vars(("temp_11_0um_nom", "temp_12_0um_nom"))
        data["observations"] = xr.concat((temp_11, temp_12), dim="channels").transpose("scans", "pixels", "channels")

        data_s = resample_and_split(
            data,
            domain[self.scale],
            time_step,
            10e3,
            include_swath_center_coords=True
        )

        if data_s is None:
            return None

        for time_ind  in range(data_s.time.size):

            data_t = data_s[{"time": time_ind}]

            comp = {
                "dtype": "uint16",
                "scale_factor": 0.01,
                "zlib": True,
                "_FillValue": 2**16 - 1,
            }
            valid_pixel_rows, valid_pixel_cols = np.where(np.any(np.isfinite(data_t.observations.data[::40, ::40]), -1))
            valid_pixel_rows = 40 * valid_pixel_rows
            valid_pixel_cols = 40 * valid_pixel_cols
            data_t["valid_pixel_cols"] = (("valid_pixels",), valid_pixel_cols)
            data_t["valid_pixel_rows"] = (("valid_pixels",), valid_pixel_rows)
            encoding = {
                "observations": comp,
                "col_inds_swath_center": {"dtype": "int16", "zlib": True},
                "row_inds_swath_center": {"dtype": "int16", "zlib": True},
                "valid_pixel_rows": {"dtype": "int16", "zlib": True},
                "valid_pixel_cols": {"dtype": "int16", "zlib": True},
            }

            filename = get_output_filename(
                self.name, data_t.time.data, time_step
            )
            output_file = output_folder / filename

            if output_file.exists():
                output_data = xr.load_dataset(output_file)
                mask = np.isfinite(data_t["observations"].data)
                output_data.observations.data[mask] = data_t["observations"].data[mask]

                mask = np.any(np.isfinite(output_data.observations.data), -1)
                valid_pixel_rows, valid_pixel_cols = np.where(mask[::40, ::40])
                valid_pixel_rows = 40 * valid_pixel_rows
                valid_pixel_cols = 40 * valid_pixel_cols
                output_data = output_data.drop_vars(("valid_pixel_rows", "valid_pixel_cols"))
                output_data["valid_pixel_rows"] = (("valid_pixels",), valid_pixel_rows)
                output_data["valid_pixel_cols"] = (("valid_pixels",), valid_pixel_cols)
                output_data.to_netcdf(output_file, encoding=encoding)
            else:
                data_t.to_netcdf(output_folder / filename, encoding=encoding)


AVHRR = AVHRRData()
