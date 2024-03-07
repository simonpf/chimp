"""
chimp.data.gridsat
=================

This module provides the GridSat class that provides an interface to extract
training data from the GridSat-B1 dataset.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from pansat import TimeRange
from pansat.products.satellite.ncei import gridsat_b1
from chimp.data.utils import scale_slices
import torch
import xarray as xr

from chimp.data.input import InputDataset


def load_gridsat_data(path):
    """
    Load GridSat observations.

    Loads GridSat visible, IR water vapor, and IR window observations and
    combines them into a single xarray.Dataset with dimensions time,
    latitude, longitude and channels.

    Args:
         path: Path object pointing to the GridSat B1 file to load.

    Return:
         An xarray.Dataset containing the loaded data.
    """
    with xr.open_dataset(path) as data:
        time = data["time"].data
        lons = data["lon"].data
        lats = data["lat"].data
        irwin = data["irwin_cdr"].data
        irwvp = data["irwvp"].data
        vschn = data["vschn"].data
    return xr.Dataset({
        "longitude": (("longitude",), lons),
        "latitude": (("latitude",), lats),
        "time": (("time",), time),
        "obs": (
            ("time", "latitude", "longitude", "channels"),
            np.stack([vschn, irwvp, irwin], -1)
        )
    })



class GridSat(InputDataset):
    """
    Provides an interface to extract and load training data from the GridSat
    B1 dataset.
    """
    def __init__(self):
        super().__init__("gridsat", "gridsat", 1, "obs", spatial_dims=("latitude", "longitude"))
        self.n_channels = 24

    def find_files():
        pass

    def process_file():
        pass

    def load_sample(
        self,
        input_file: Path,
        crop_size: Union[int, Tuple[int, int]],
        base_scale: int,
        slices: Tuple[slice, slice],
        rng: np.random.Generator,
        missing_value_policy: str,
        rotate: Optional[float] = None,
        flip: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Load input data sample from file.

        Args:
            input_file: Path pointing to the input file from which to load the
                data.
            crop_size: Size of the final crop.
            base_scale: The scale of the reference data.
            sclices: Tuple of slices defining the part of the data to load.
            rng: A numpy random generator object to use to generate random
                data.
            missing_value_policy: A string describing how to handle missing
                values.
            rotate: If given, the should specify the number of degree by
                which the input should be rotated.
            flip: Bool indicated whether or not to flip the input along the
                last dimensions.

        Return:
            A torch tensor containing the loaded input data.
        """
        rel_scale = self.scale / base_scale

        if isinstance(crop_size, int):
            crop_size = (crop_size,) * self.n_dims
        crop_size = tuple((int(size / rel_scale) for size in crop_size))

        if input_file is not None:
            slices = scale_slices(slices, rel_scale)
            with xr.open_dataset(input_file) as data:

                if data.time.size < 8:
                    ttype = data.time.data.dtype
                    start_time = data.time.data[0].astype("datetime64[D]").astype(ttype)
                    end_time = start_time + np.timedelta64(1, "D")
                    time = np.arange(start_time, end_time, np.timedelta64(3, "h"))
                    data = data.interp(time=time, method="nearest")

                vars = self.variables
                if not isinstance(vars, list):
                    vars = [vars]
                all_data = []
                for vrbl in vars:
                    x_s = data[vrbl][dict(zip(self.spatial_dims, slices))].data
                    if x_s.ndim < 3:
                        x_s = x_s[None]
                    x_s = np.transpose(x_s, (0, 3, 1, 2))
                    x_s = x_s.reshape((-1,) + x_s.shape[2:])
                    all_data.append(x_s)
                x_s = np.concatenate(all_data, axis=0)

            # Apply augmentations
            if rotate is not None:
                x_s = ndimage.rotate(
                    x_s, rotate, order=0, reshape=False, axes=(-2, -1), cval=np.nan
                )
                height = x_s.shape[-2]

                # In case of a rotation, we may need to cut off some input.
                height_out = crop_size[0]
                if height > height_out:
                    start = (height - height_out) // 2
                    end = start + height_out
                    x_s = x_s[..., start:end, :]

                width = x_s.shape[-1]
                width_out = crop_size[1]
                if width > width_out:
                    start = (width - width_out) // 2
                    end = start + width_out
                    x_s = x_s[..., start:end]
            if flip:
                x_s = np.flip(x_s, -2)
        else:
            if missing_value_policy == "sparse":
                return None
            x_s = np.nan * np.ones(((self.n_channels,) + crop_size), dtype=np.float32)

        # If we are here we're not returning None.
        if missing_value_policy == "sparse":
            missing_value_policy = "missing"

        x_s = torch.tensor(x_s.copy(), dtype=torch.float32)
        if missing_value_policy == "masked":
            mask = torch.ones_like(x_s).to(dtype=bool)
            x_s = MaskedTensor(x_s, mask=mask)
        x_s = self.replace_missing(x_s, missing_value_policy, rng)

        return x_s

    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(days=1),
            include_scan_time=False
    ):
        """
        Extract training data for a given day.

        Args:
            domain: A domain object identifying the spatial domain for which
                to extract input data.
            year: The year
            month: The month
            day: The day
            output_folder: The folder to which to write the extracted
                observations.
            path: Not used, included for compatibility.
            time_step: The temporal resolution of the training data.
            include_scan_time: Not used.
        """
        output_folder = Path(output_folder) / "gridsat"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        time = datetime(year=year, month=month, day=day)
        end = time + timedelta(days=1)

        if isinstance(domain, dict):
            domain = domain[8]
        lons, lats = domain.get_lonlats()
        lons = lons[0]
        lats = lats[..., 0]

        while time < end:
            if time_step.total_seconds() > 3 * 60 * 60:
                time_range = TimeRange(
                    time,
                    time + time_step - timedelta(hours=1, minutes=30, seconds=1)
                )
            else:
                time_range = TimeRange(
                    time,
                    time + time_step - timedelta(seconds=1)
                )

            recs = gridsat_b1.find_files(time_range)
            recs = [rec.get() for rec in recs]
            gridsat_data = xr.concat(
                [load_gridsat_data(rec.local_path) for rec in recs],
                dim="time"
            )
            gridsat_data = gridsat_data.interp(
                latitude=lats,
                longitude=lons
            )
            if time_step.total_seconds() < 3 * 60 * 60:
                gridsat_data = gridsat_data.interp(time=time)

            filename = time.strftime("gridsat_%Y%m%d_%H_%M.nc")

            encodings = {
                obs: {"dtype": "float32", "zlib": True}
                for obs in gridsat_data.variables
            }
            gridsat_data.to_netcdf(output_folder / filename, encoding=encodings)

            time = time + time_step


gridsat = GridSat()
