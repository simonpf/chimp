"""
chimp.data.wxfm
===============

This module provides input data for the NASA Weather and Climate Foundation
Model (WxFM).
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from pansat import FileRecord, Geometry, TimeRange
from pansat.products.reanalysis.merra import MERRA2, MERRA2Constant
import xarray as xr

from chimp.areas import Area
from chimp.data import InputDataset
from chimp.data.utils import (
    get_output_filename,
    records_to_paths,
    round_time
)


LOGGER = logging.getLogger(__name__)


LEVELS = [
    72.0, 71.0, 68.0, 63.0, 56.0, 53.0, 51.0, 48.0, 45.0, 44.0, 43.0, 41.0, 39.0, 34.0
]
VARIABLES_ATMOS = [
    "U", "V", "OMEGA", "T", "QV", "PL", "H", "CLOUD", "QI", "QL", "U10",
    "V10", "T2M", "QV2M", "PS", "TS", "TQI", "TQL", "TQV", "GWETROOT",
    "LAI", "EFLUX", "HFLUX", "PRECTOT", "Z0M", "LWGEM", "LWGAB", "LWTUP",
    "SWGNT", "SWTNT"
]
SURFACE_VARS = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "Z0M"
]

STATIC_SURFACE_VARS = [
    "FRACI",
    "FRLAND",
    "FROCEAN",
    "PHIS"
]

VERTICAL_VARS = [
    "CLOUD",
    "H",
    "OMEGA",
    "PL",
    "QI",
    "QL",
    "QV",
    "T",
    "U",
    "V"
]

m2i3nwasm = MERRA2(
    collection="m2i3nvasm",
)
m2i1nxasm = MERRA2(
    collection="m2i1nxasm",
)
m2t1nxlnd = MERRA2(
    collection="m2t1nxlnd",
    variables=[
        "GWETROOT", "LAI"
    ]
)
m2t1nxflx = MERRA2(
    collection="m2t1nxflx",
)
m2t1nxrad = MERRA2(
    collection="m2t1nxrad",
)

DYNAMIC_PRODUCTS = [
    m2i3nwasm,
    m2i1nxasm,
    m2t1nxlnd,
    m2t1nxflx,
    m2t1nxrad
]



class WxFMDynamicData(InputDataset):
    """
    Represents dynamic input data for the NASA WxFM model.
    """
    def __init__(self):
        InputDataset.__init__(self, "wxfm_dynamic", "wxfm_dynamic", 64, "input", n_dim=2)
        self.products = DYNAMIC_PRODUCTS


    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[List[FileRecord]]:
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
        matches = {}
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.nc4")))
            matching = []
            for prod in self.products:
                if not prod.matches(path):
                    continue
                time_range = prod.get_temporal_coverage(path)
                start_time = time_range.start
                if prod.get_temporal_coverage(path).covers(time_range):
                    matches.setdefault(start_time, []).append(prod)
        else:
            for prod in self.products:
                recs = prod.find_files(TimeRange(start_time, end_time))

                for rec in recs:
                    start_time = rec.temporal_coverage.start
                    matches.setdefault(start_time, []).append(rec)

        return list(matches.values())


    def process_file(
            self,
            recs: List[Path | FileRecord],
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given input data file.

        Args:
            paths: A list containing the file records to process.
            domain: An area object defining the training domain.
            output_folder: A path pointing to the folder to which to write
                the extracted training data.
            time_step: A timedelta object defining the retrieval time step.
        """
        assert len(recs) == len(self.products)
        paths = records_to_paths(recs)

        all_data = []
        for path in paths:
            with xr.open_dataset(path) as data:
                vars = [
                    var for var in VERTICAL_VARS + SURFACE_VARS if var in data.variables
                ]
                data = data[vars]
                if "lev" in data:
                    data = data.loc[{"lev": np.array(LEVELS)}]
                all_data.append(data.load())
        data = xr.merge(all_data)
        data = data.rename(
            lat="latitude",
            lon="longitude"
        )

        lons, lats = domain[self.scale].get_lonlats()
        if np.all(lons[0] == lons[1]) and np.all(lats[:, 0] == lats[:, 1]):
            lons = lons[0]
            lats = lats[:, 0]
            data = data.interp(latitude=lats, longitude=lons, method="linear")
        else:
            lons = xr.DataArray(lons, dims=("y", "x"))
            lats = xr.DataArray(lats, dims=("y", "x"))
            data = data.interp(latitude=lats, longitude=lons)

        output_path = Path(output_folder) / "wxfm_dynamic"
        output_path.mkdir(exist_ok=True, parents=True)

        start_time = round_time(data.time.min().data, time_step)
        end_time = round_time(data.time.max().data, time_step)
        time = start_time
        while time <= end_time:
            data_t = data.interp(time=time, method="nearest")
            filename = get_output_filename(
                self.name, time, time_step
            )
            encoding = {
                var: {"dtype": "float32", "zlib": True} for var in data_t.variables
            }
            data_t.to_netcdf(output_path / filename, encoding=encoding)
            time += time_step



WXFM_DYNAMIC = WxFMDynamicData()

m2conxasm = MERRA2Constant(
    collection="m2conxasm",
)
m2conxctm = MERRA2Constant(
    collection="m2conxctm",
)
STATIC_PRODUCTS = [
    m2conxasm,
    m2conxctm
]

class WxFMStaticData(InputDataset):
    """
    Represents input data for the NASA WxFM model.
    """
    def __init__(self):
        InputDataset.__init__(self, "wxfm_static", "wxfm_static", 64, "input", n_dim=2)
        self.products = STATIC_PRODUCTS


    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[List[FileRecord]]:
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
        recs = []
        for prod in self.products:
            recs += prod.find_files(TimeRange("2000-01-01"))
        return [recs]


    def process_file(
            self,
            recs: List[Path | FileRecord],
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given input data file.

        Args:
            paths: A list containing the file records to process.
            domain: An area object defining the training domain.
            output_folder: A path pointing to the folder to which to write
                the extracted training data.
            time_step: A timedelta object defining the retrieval time step.
        """
        assert len(recs) == len(self.products)
        paths = records_to_paths(recs)

        all_data = []
        for path in paths:
            with xr.open_dataset(path) as data:
                if "lev" in data:
                    data = data.loc[{"lev": LEVELS}]
                all_data.append(data.load())
        data = xr.merge(all_data)
        data = data.rename(
            lat="latitude",
            lon="longitude"
        )

        lons, lats = domain[self.scale].get_lonlats()
        if np.all(lons[0] == lons[1]) and np.all(lats[:, 0] == lats[:, 1]):
            lons = lons[0]
            lats = lats[:, 0]
            data = data.interp(latitude=lats, longitude=lons, method="linear")
        else:
            lons = xr.DataArray(lons, dims=("y", "x"))
            lats = xr.DataArray(lats, dims=("y", "x"))
            data = data.interp(latitude=lats, longitude=lons)

        output_path = Path(output_folder) / "wxfm_static"
        output_path.mkdir(exist_ok=True, parents=True)
        filename = self.name + ".nc"
        encoding = {
            var: {"dtype": "float32", "zlib": True} for var in data.variables
        }
        data.to_netcdf(output_path / filename, encoding=encoding)


WXFM_STATIC = WxFMStaticData()
