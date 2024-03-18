"""
chimp.data.mrms
==============

This module implements the functionality to download and resample
MRMS surface precipitation data to be used as reference data for
training CHIMP retrievals.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
from scipy.stats import binned_statistic_2d
from pansat import FileRecord
from pansat.geometry import Geometry
from pansat.download.providers import IowaStateProvider
from pansat.products.ground_based import mrms
from pansat.utils import resample_data
from pansat.time import TimeRange

from pansat.time import to_datetime64
import xarray as xr

from chimp.areas import Area
from chimp.data.utils import get_output_filename, round_time, records_to_paths
from chimp.data.reference import ReferenceDataset, RetrievalTarget


LOGGER = logging.getLogger(__name__)


PRECIP_TYPES = {
    "No rain": [0.0],
    "Stratiform": [1.0, 2.0, 10.0, 91.0],
    "Convective": [6.0, 96.0],
    "Hail": [7.0],
    "Snow": [3.0, 4.0],
}


def save_file(dataset, output_folder, filename):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the MRMS observations.
        output_folder: The folder to which to write the training data.

    """
    output_filename = Path(output_folder) / filename

    if output_filename.exists():
        output_data = xr.load_dataset(output_filename)
    else:
        output_data = xr.Dataset()

    comp = {"zlib": True}
    encoding = {var: comp for var in output_data.variables.keys()}

    if "surface_precip" in dataset:
        surface_precip = np.minimum(dataset.surface_precip.data, 300)
        dataset.surface_precip.data = surface_precip
        encoding["surface_precip"] = {
            "scale_factor": 1 / 100,
            "dtype": "int16",
            "zlib": True,
            "_FillValue": -1
        }
        output_data["surface_precip"] = (("y", "x"), surface_precip)

    if "rqi" in dataset:
        encoding["rqi"] = {
            "scale_factor": 1 / 127,
            "dtype": "int8",
            "zlib": True,
            "_FillValue": -1
        }
        output_data["rqi"] = (("y", "x"), dataset.rqi.data)

    if "precip_type" in dataset:
        encoding["precip_type"] = {
            "dtype": "uint8",
            "zlib": True
        }
        output_data["precip_type"] = (("y", "x"), dataset.precip_type.data.astype("int8"))

    output_data.to_netcdf(output_filename, encoding=encoding)


class MRMSData(ReferenceDataset):
    """
    Represents reference data derived from MRMS ground-based radar measurements.
    """
    def __init__(
            self,
            name: str,
            scale: int,
            targets: List[RetrievalTarget],
            quality_index: str
    ):
        """
        Args:
            name: The name of the reference dataset.
            scale: The spatial scale of the data.
            targets: A list of the retrieval targets provided by the dataset.
            quality_index: Name of the field hodling the quality index.
        """
        super().__init__(name, scale, targets, quality_index)
        self.products = [mrms.precip_rate, mrms.radar_quality_index, mrms.precip_flag]


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
        found_files = {}

        if path is not None:
            all_files = sorted(list(path.glob("**/*.grib2.gz")))

        for prod in self.products:
            if path is not None:
                recs = [
                    FileRecord(prod, path) for path in all_files
                    if prod.matches(path)
                ]
            else:
                recs = prod.find_files(TimeRange(start_time, end_time))

            matched_recs = {}
            matched_deltas = {}

            for rec in recs:
                tr_rec = rec.temporal_coverage
                time_c = to_datetime64(tr_rec.start + 0.5 * (tr_rec.end - tr_rec.start))
                time_n = round_time(time_c, time_step)
                delta = abs(time_c - time_n)

                min_delta = matched_deltas.get(time_n, None)
                if min_delta is None:
                    matched_recs[time_n] = rec
                    matched_deltas[time_n] = delta
                else:
                    if delta < min_delta:
                        matched_recs[time_n] = rec
                        matched_deltas[time_n] = delta

            for time_n, matched_rec in matched_recs.items():
                found_files.setdefault(time_n, []).append(matched_rec)

        return list(found_files.values())



    def process_file(
        self,
        path: Path,
        domain: Area,
        output_folder: Path,
        time_step: np.timedelta64
    ):
        if not isinstance(path, list):
            paths = [path]
        else:
            paths = path
        paths = records_to_paths(paths)

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        data = []
        for path in paths:
            product = [prod for prod in self.products if prod.matches(path)][0]
            data.append(product.open(path))
        data = xr.merge(data)
        new_names = {
            "precip_rate": "surface_precip",
            "radar_quality_index": "rqi",
            "precip_flag": "precip_type"
        }
        new_names = {
            name: new_names[name] for name in data.variables if name in new_names
        }
        data = data.rename(new_names)
        data = resample_data(data, domain[4], radius_of_influence=4e3, new_dims=("y", "x"))
        if "precip_type" in data:
            precip_type = np.nan_to_num(data.precip_type.data, nan=-1, copy=True).astype(np.int8)
            data["precip_type"] = (data.surface_precip.dims, precip_type)
        time_range = product.get_temporal_coverage(path)
        time_c = time_range.start + 0.5 * (time_range.end - time_range.start)
        filename = get_output_filename("mrms", time_c, time_step)
        save_file(data, output_folder, filename)


    def find_training_files(self, path: Path) -> List[Path]:
        """
        Find MRMS training data files.

        Args:
            path: Path to the folder containing the training data.

        Return:
            A list of found reference data files.
        """
        pattern = "*????????_??_??.nc"
        reference_files = sorted(
            list((path / "mrms").glob(pattern))
        )
        return reference_files




MRMS_PRECIP_RATE = MRMSData(
    "mrms",
    4,
    [RetrievalTarget("surface_precip", 1e-3)],
    "rqi"
)

MRMS_PRECIP_RATE_AND_TYPE = MRMSData(
    "mrms_w_type",
    4,
    [
        RetrievalTarget("surface_precip", 1e-3),
        RetrievalTarget("precip_type", None),
    ],
    "rqi"
)
