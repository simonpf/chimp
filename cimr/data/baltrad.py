"""
cimr.data.baltrad
=================

Functionality to simplify reading and processing of Baltrad
data.
"""
from datetime import datetime
from pathlib import Path
import re

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
import xarray as xr

from cimr.utils import round_time

class Baltrad:
    """
    Interface class to read Baltrad data.
    """
    @staticmethod
    def filename_to_date(filename):
        """
        Extract time from filename.

        Args:
            filename: The name or path to the file.

        Return
            ``np.datetime64`` object representing the time.
        """
        name = Path(filename).name
        date = name.split("_")[4]
        date = datetime.strptime(date, "%Y%m%dT%H%M%SZ")
        return pd.Timestamp(date).to_datetime64()

    @staticmethod
    def find_files(base_dir, start_time=None, end_time=None):
        """
        Find Baltrad files.

        Args:
            base_dir: Root directory to search through.
            start_time: Optional start time to limit the search.
            end_time: Optional end time to limit the search.

        Return:
            A list of the found files.
        """
        files = list(Path(base_dir).glob("**/comp_pcappi_blt2km_pn150*.h5"))
        dates = np.array(list((map(Baltrad.filename_to_date, files))))

        if len(dates) == 0:
            return []

        if start_time is None:
            start_time = dates.min()
        else:
            start_time = np.datetime64(start_time)

        if end_time is None:
            end_time = dates.max()
        else:
            end_time = np.datetime64(end_time)

        return [
            file for file, date in zip(files, dates)
            if (date >= start_time) and (date <= end_time)
        ]

        return files

    def __init__(self, filename):
        """
        Create Baltrad file but don't load the data yet.

        Args:
            filename: Path to the file containing the Baltrad data.
        """
        self.filename = filename
        self._load_projection_info()

    def _load_projection_info(self):
        """
        Loads the projection info from the Baltrad file.
        """
        with File(self.filename, "r") as data:

            projdef = data["where"].attrs["projdef"].decode() + " +units=m"
            size_x = data["where"].attrs["xsize"]
            size_y = data["where"].attrs["ysize"]

            latlon = "+proj=longlat +ellps=bessel +datum=WGS84 +units=m"

            transformer = Transformer.from_crs(
                latlon,
                projdef,
                always_xy=True
            )

            lon_min = data["where"].attrs["LL_lon"]
            lat_min = data["where"].attrs["LL_lat"]
            lon_max = data["where"].attrs["UR_lon"]
            lat_max = data["where"].attrs["UR_lat"]

            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            area = AreaDefinition(
                "CIMR_NORDIC",
                "CIMR region of interest over the nordic countries.",
                "CIMR_NORDIC",
                projdef,
                size_x,
                size_y,
                (x_min, y_min, x_max, y_max)
            )
            n_rows, n_cols = area.shape
            new_shape = ((n_rows // 4) * 4, (n_cols // 4) * 4)
            area = area[(slice(0, new_shape[0]), slice(0, new_shape[1]))]

            self.area = area

    def to_xarray_dataset(self):
        """
        Load data from file into xarray dataset.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        with File(self.filename, "r") as data:

            i_end, j_end = self.area.shape

            #
            # DBZ
            #

            dbz = data["dataset1/data3"]["data"][0:i_end, 0:j_end]
            dataset = xr.Dataset({
                "dbz": (("y", "x"), dbz)
            })

            gain = data["dataset1"]["data3"]["what"].attrs["gain"]
            offset = data["dataset1"]["data3"]["what"].attrs["offset"]
            no_data = data["dataset1"]["data3"]["what"].attrs["nodata"]
            undetect = data["dataset1"]["data3"]["what"].attrs["undetect"]
            qty = data["dataset1"]["data3"]["what"].attrs["quantity"].decode()

            dataset.dbz.attrs["scale_factor"] = gain
            dataset.dbz.attrs["add_offset"] = offset
            dataset.dbz.attrs["missing_value"] = no_data
            dataset.dbz.attrs["undetect"] = undetect
            dataset.dbz.attrs["quantity"] = qty

            #
            # Quality index
            #

            qi = data["dataset1/data3/quality4"]["data"][0:i_end, 0:j_end]
            dataset["qi"] = (("y", "x"), qi)

            gain = data["dataset1"]["data3"]["quality4"]["what"].attrs["gain"]
            offset = data["dataset1"]["data3"]["quality4"]["what"].attrs["offset"]

            dataset.qi.attrs["scale_factor"] = gain
            dataset.qi.attrs["add_offset"] = offset

            start_date = data["dataset1"]["what"].attrs["startdate"].decode()
            start_time = data["dataset1"]["what"].attrs["starttime"].decode()
            dataset.attrs["start_time"] = f"{start_date} {start_time}"

            end_date = data["dataset1"]["what"].attrs["enddate"].decode()
            end_time = data["dataset1"]["what"].attrs["endtime"].decode()
            dataset.attrs["end_time"] = f"{end_date} {end_time}"

            time = Baltrad.filename_to_date(self.filename)
            dataset.attrs["time"] = str(time)
            dataset.attrs["projection"] = self.area.proj_str

        return dataset


def save_file(dataset, output_folder):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the resampled
            Baltrad observations.
        output_folder: The folder to which to write the training data.

    """
    time_15 = round_time(dataset.attrs["time"])
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"radar_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename
    dataset.to_netcdf(output_filename)
