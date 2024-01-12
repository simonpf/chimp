"""
chimp.data.baltrad
=================

Defines the Baltrad input data class that provides an interface to extract
and load precipitation estimates from the BALTRAD radar network.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Optional

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
import xarray as xr

from pansat import FileRecord, TimeRange, Geometry
from pansat.geometry import LonLatRect
from pansat.catalog import Index
from pansat.products import Product, FilenameRegexpMixin

from chimp.data import ReferenceData
from chimp.data.reference import RetrievalTarget
from chimp.utils import round_time
from chimp.data.resample import resample_data
from chimp.data.utils import get_output_filename


def _load_projection_info(path):
    """
    Loads the projection info from the Baltrad file.
    """
    with File(path, "r") as data:
        projdef = data["where"].attrs["projdef"].decode() + " +units=m"
        size_x = data["where"].attrs["xsize"]
        size_y = data["where"].attrs["ysize"]

        latlon = "+proj=longlat +ellps=bessel +datum=WGS84 +units=m"

        transformer = Transformer.from_crs(latlon, projdef, always_xy=True)

        lon_min = data["where"].attrs["LL_lon"]
        lat_min = data["where"].attrs["LL_lat"]
        lon_max = data["where"].attrs["UR_lon"]
        lat_max = data["where"].attrs["UR_lat"]

        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)

        area = AreaDefinition(
            "CHIMP_NORDIC",
            "CHIMP region of interest over the nordic countries.",
            "CHIMP_NORDIC",
            projdef,
            size_x,
            size_y,
            (x_min, y_min, x_max, y_max),
        )
    return area


class BaltradData(FilenameRegexpMixin, Product):
    """
    pansat product class to access BALTRAD data.
    """

    def __init__(self):
        self.filename_regexp = re.compile(
            r"comp_pcappi_blt2km_pn150_(?P<date>\d{8}T\d{6})Z_(\w*).h5"
        )
        Product.__init__(self)

    @property
    def name(self) -> str:
        return "chimp.baltrad"

    @property
    def default_destination(self) -> str:
        return "baltrad"

    def get_temporal_coverage(self, rec: FileRecord) -> TimeRange:
        """
        Return temporal coverage of file.

        Args:
            rec: A file record pointing to a Baltrad file.

        Return:
            A TimeRange object representing the temporal coverage of
            the file identified by 'rec'.
        """
        if not isinstance(rec, FileRecord):
            rec = FileRecord(local_path=rec)

        match = self.filename_regexp.match(rec.local_path.name)
        if match is None:
            raise ValueError(
                "The provided file record does not point to a BALTRAD file."
            )
        start = datetime.strptime(match.group("date"), "%Y%m%dT%H%M%S")
        end = start + timedelta(minutes=15)
        return TimeRange(start, end)

    def get_spatial_coverage(self, rec: FileRecord) -> Geometry:
        """
        Return spatial coverage of file.

        Args:
            rec: A file record pointing to a Baltrad file.

        Return:
            A pansat.Geometry object representing the spatial coverage of the
            file identified by 'rec'.
        """
        return LonLatRect(-180, -90, 180, 90)

    def open(self, rec: FileRecord) -> xr.Dataset:
        """
        Load data from Baltrad file.

        Args:
            rec: A file record whose 'local_path' attributes points to a
                local file to load.

        Return:
            An ``xarray.Dataset`` containing the data from the file.
        """
        if not isinstance(rec, FileRecord):
            rec = FileRecord(rec)

        with File(rec.local_path, "r") as data:

            #
            # DBZ
            #

            dbz = np.array(data["dataset1/data3"]["data"][:])
            gain = data["dataset1"]["data3"]["what"].attrs["gain"]
            offset = data["dataset1"]["data3"]["what"].attrs["offset"]
            no_data = data["dataset1"]["data3"]["what"].attrs["nodata"]
            undetect = data["dataset1"]["data3"]["what"].attrs["undetect"]
            qty = data["dataset1"]["data3"]["what"].attrs["quantity"].decode()

            dbz = dbz.astype(np.float32)
            dbz[dbz == no_data] = np.nan
            dbz[dbz == undetect] = np.nan
            dbz = dbz * gain + offset
            dataset = xr.Dataset({"dbz": (("y", "x"), dbz)})

            dataset.dbz.attrs["scale_factor"] = gain
            dataset.dbz.attrs["add_offset"] = offset
            dataset.dbz.attrs["missing"] = no_data
            dataset.dbz.attrs["undetect"] = undetect
            dataset.dbz.attrs["quantity"] = qty

            #
            # Quality index
            #

            qi = np.array(data["dataset1/data3/quality4"]["data"][:])
            gain = data["dataset1"]["data3"]["quality4"]["what"].attrs["gain"]
            offset = data["dataset1"]["data3"]["quality4"]["what"].attrs["offset"]
            qi = qi.astype(np.float32)
            qi = qi * gain + offset
            dataset["qi"] = (("y", "x"), qi)


            dataset.qi.attrs["scale_factor"] = gain
            dataset.qi.attrs["add_offset"] = offset

            start_date = data["dataset1"]["what"].attrs["startdate"].decode()
            start_time = data["dataset1"]["what"].attrs["starttime"].decode()
            dataset.attrs["start_time"] = f"{start_date} {start_time}"

            end_date = data["dataset1"]["what"].attrs["enddate"].decode()
            end_time = data["dataset1"]["what"].attrs["endtime"].decode()
            dataset.attrs["end_time"] = f"{end_date} {end_time}"

            time = self.get_temporal_coverage(rec)
            dataset.attrs["time"] = str(time.start)
            area = _load_projection_info(rec.local_path)
            dataset.attrs["projection"] = area

            lons, lats = area.get_lonlats()
            dataset["longitude"] = (("y", "x"), lons)
            dataset["latitude"] = (("y", "x"), lats)

        return dataset


baltrad_product = BaltradData()


class Baltrad(ReferenceData):
    """
    The Baltrad input data class that extracts radar reflectivity and
    estimates from BALTRAD files.
    """

    def __init__(self):
        super().__init__(
            "baltrad", scale=4, targets=[RetrievalTarget("dbz")], quality_index="qi"
        )

    def process_day(
        self,
        domain: dict,
        year: int,
        month: int,
        day: int,
        output_folder: Path,
        path: Path = Optional[None],
        time_step: timedelta = timedelta(minutes=15),
        include_scan_time=False,
    ):
        """
        Extract GMI input observations for the CHIMP retrieval.

        Args:
            domain: A domain dict specifying the area for which to
                extract GMI input data.
            year: The year.
            month: The month.
            day: The day.
            output_folder: The root of the directory tree to which to write
                the training data.
            path: Not used, included for compatibility.
            time_step: The time step between consecutive retrieval steps.
            include_scan_time: If set to 'True', the resampled scan time will
                be included in the extracted training input.
        """
        output_folder = Path(output_folder) / self.name
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        files = (path / f"{year}/{month:02}/{day:02}").glob("**/*.h5")
        index = Index.index(baltrad_product, files)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
        time_range = TimeRange(start_time, end_time)

        time = start_time

        if isinstance(domain, dict):
            domain = domain[self.scale]

        while time < end_time:
            print(time)
            granules = index.find(
                TimeRange(
                    time - 0.5 * time_step,
                    time + 0.5 * time_step,
                )
            )
            if len(granules) > 0:
                data = baltrad_product.open(granules[0].file_record)
                data_r = resample_data(data, domain, radius_of_influence=4e3)
                data_r = data_r.drop_vars(["latitude", "longitude"])

                output_filename = get_output_filename(
                    "baltrad", time, minutes=time_step.total_seconds() // 60
                )

                encoding = {
                    "dbz": {
                        "add_offset": data.dbz.attrs["add_offset"],
                        "scale_factor": data.dbz.attrs["scale_factor"],
                        "_FillValue": data.dbz.attrs["missing"],
                        "zlib": True
                    },
                    "qi": {
                        "add_offset": data.qi.attrs["add_offset"],
                        "scale_factor": data.qi.attrs["scale_factor"],
                        "zlib": True
                    }
                }

                data_r.to_netcdf(output_folder / output_filename, encoding=encoding)

            time = time + time_step


baltrad = Baltrad()
