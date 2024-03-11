"""
chimp.data.baltrad
=================

Defines the Baltrad input data class that provides an interface to extract
and load precipitation estimates from the BALTRAD radar network.
"""
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional

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
from pansat.time import to_datetime64

from chimp.areas import Area
from chimp.data import ReferenceDataset
from chimp.data.reference import RetrievalTarget
from chimp.data.resample import resample_and_split
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
            dataset["time"] = (("time"), np.array([to_datetime64(time.start)]))
            area = _load_projection_info(rec.local_path)
            dataset.attrs["projection"] = area

            lons, lats = area.get_lonlats()
            dataset["longitude"] = (("y", "x"), lons)
            dataset["latitude"] = (("y", "x"), lats)

        return dataset


baltrad_product = BaltradData()


class Baltrad(ReferenceDataset):
    """
    The Baltrad input data class that extracts radar reflectivity and
    estimates from BALTRAD files.
    """

    def __init__(self):
        super().__init__(
            "baltrad", scale=4, targets=[RetrievalTarget("dbz")], quality_index="qi"
        )


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
        if path is None:
            raise RuntimeError(
                "BALTRAD data is not available publicly. Therefore, the 'path' argument"
                " must not be 'None'."
            )
        path = Path(path)
        all_files = sorted(list(path.glob("**/*.h5")))
        matching = [path for path in all_files if baltrad_product.matches(path)]
        return matching


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


        time_range = baltrad_product.get_temporal_coverage(path)
        time = time_range.start

        data = baltrad_product.open(path)
        data_r = resample_and_split(data, domain[self.scale], time_step=time_step, radius_of_influence=4e3)
        data_r = data_r[{"time": 0}]
        data_r = data_r.drop_vars(["latitude", "longitude"])

        output_filename = get_output_filename(
            "baltrad", time, time_step
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


BALTRAD = Baltrad()
