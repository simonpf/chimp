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
from typing import List

from h5py import File
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyresample.geometry import AreaDefinition
from scipy.stats import binned_statistic_2d
from pansat.download.providers import IowaStateProvider
from pansat.products.ground_based import mrms
from pansat.utils import resample_data
from pansat.time import TimeRange

from pansat.time import to_datetime64
import xarray as xr

from chimp.utils import round_time
from chimp.data.utils import get_output_filename
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp import areas


LOGGER = logging.getLogger(__name__)

PRECIP_TYPES = {
    "No rain": [0.0],
    "Stratiform": [1.0, 2.0, 10.0, 91.0],
    "Convective": [6.0, 96.0],
    "Hail": [7.0],
    "Snow": [3.0, 4.0],
}


def resample_mrms_data(
        dataset: xr.Dataset,
        domain: areas.Area
) -> xr.Dataset:
    """
    Resample MRMS data to CHIMP grid.

    Args:
        dataset: 'xarray.Dataset' containing the MRMS surface precip and
            radar quality index.
        area: An Area object defining the domain to which to resample the MRMS
            data.

    Return:
        A new dataset containing the surface precip and RQI data
        resampled to the CONUS 4-km domain.
    """
    lons_out, lats_out = domain[4].get_lonlats()

    lons_out = lons_out[0]
    lon_bins = np.zeros(lons_out.size + 1)
    lon_bins[1:-1] = 0.5 * (lons_out[1:] + lons_out[:-1])
    lon_bins[0] = lon_bins[1] - (lon_bins[2] - lon_bins[1])
    lon_bins[-1] = lon_bins[-2] + lon_bins[-2] - lon_bins[-3]

    lats_out = lats_out[..., 0][::-1]
    lat_bins = np.zeros(lats_out.size + 1)
    lat_bins[1:-1] = 0.5 * (lats_out[1:] + lats_out[:-1])
    lat_bins[0] = lat_bins[1] - (lat_bins[2] - lat_bins[1])
    lat_bins[-1] = lat_bins[-2] + lat_bins[-2] - lat_bins[-3]

    flipped = False
    print(np.diff(lat_bins))
    if np.diff(lat_bins)[0] < 0:
        flipped = True
        lat_bins = lat_bins[::-1]

    lons, lats = areas.MRMS[4].get_lonlats()
    sfp = dataset.surface_precip.data
    valid = sfp >= 0.0
    surface_precip = binned_statistic_2d(
        lats[valid],
        lons[valid],
        sfp[valid],
        bins=(lat_bins, lon_bins)
    )[0][::-1]
    rqi = dataset.rqi.data
    valid = rqi >= 0.0
    rqi = binned_statistic_2d(
        lats[valid],
        lons[valid],
        rqi[valid],
        bins=(lat_bins, lon_bins)
    )[0][::-1]

    precip_flag = dataset.precip_type.data
    precip_type_cts = []
    names = []
    for name, flags in PRECIP_TYPES.items():
        type_field = None
        for type_flag in flags:
            if type_field is None:
                type_field = np.isclose(precip_flag, type_flag)
            else:
                type_field += np.isclose(precip_flag, type_flag)
        cts = binned_statistic_2d(
            lats[valid],
            lons[valid],
            type_field[valid],
            bins=(lat_bins, lon_bins)
        )[0][::-1]
        precip_type_cts.append(cts)
    precip_type_cts = np.argmax(np.stack(precip_type_cts), axis=0)
    precip_type_cts = np.nan_to_num(precip_type_cts, nan=-1).astype("int8")

    if flipped:
        surface_precip = surface_precip[::-1]
        rqi = rqi[::-1]
        precip_type_cts = precip_type_cts[::-1]


    dataset_r = xr.Dataset({
        "latitude": (("latitude"), lats_out),
        "longitude": (("longitude"), lons_out),
        "surface_precip": (("latitude", "longitude"), surface_precip),
        "rqi": (("latitude", "longitude"), rqi),
        "precip_type": (("latitude", "longitude"), precip_type_cts),
        "time": ((), dataset.time.data)
    })
    return dataset_r


def save_file(dataset, output_folder, filename):
    """
    Save file to training data.

    Args:
        dataset: The ``xarray.Dataset`` containing the MRMS observations.
        output_folder: The folder to which to write the training data.

    """
    output_filename = Path(output_folder) / filename

    comp = {"zlib": True}
    encoding = {var: comp for var in dataset.variables.keys()}

    surface_precip = np.minimum(dataset.surface_precip.data, 300)
    dataset.surface_precip.data = surface_precip
    encoding["surface_precip"] = {
        "scale_factor": 1 / 100,
        "dtype": "int16",
        "zlib": True,
        "_FillValue": -1
    }

    dataset.surface_precip.data = surface_precip
    encoding["rqi"] = {
        "scale_factor": 1 / 127,
        "dtype": "int8",
        "zlib": True,
        "_FillValue": -1
    }

    encoding["precip_type"] = {
        "dtype": "uint8",
        "zlib": True
    }

    dataset.to_netcdf(output_filename, encoding=encoding)


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

    def find_files():
        pass

    def process_file():
        pass

    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(minutes=15),
            include_scan_time=False
    ):
        """
        Extract training data from a day of MRMS measurements.

        Args:
            year: The year
            month: The month
            day: The day
            output_folder: The folder to which to write the extracted
                observations.
            path: Not used, included for compatibility.
            time_step: Time step defining the temporal resolution at which to extract
                training samples.
            include_scan_time: Ignored. Included for compatibility.
        """
        output_folder = Path(output_folder) / "mrms"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=24)
        time = start_time
        files = []

        while time < end_time:

            time_range = TimeRange(time, time)

            precip_rate_rec = mrms.precip_rate.get(time_range)[0]
            rqi_rec = mrms.radar_quality_index.get(time_range)[0]
            precip_flag_rec = mrms.precip_flag.get(time_range)[0]

            precip_rate_data = mrms.precip_rate.open(precip_rate_rec)
            rqi_data = mrms.radar_quality_index.open(rqi_rec)
            precip_flag_data = mrms.precip_flag.open(precip_flag_rec)

            dataset = xr.merge(
                [precip_rate_data, rqi_data, precip_flag_data],
                compat="override"
            )
            dataset = dataset.rename({
                "precip_rate": "surface_precip",
                "radar_quality_index": "rqi",
                "precip_flag": "precip_type"
            })
            dataset = resample_data(dataset, domain[4], radius_of_influence=4e3, new_dims=("y", "x"))
            filename = get_output_filename("mrms", time, time_step)
            save_file(dataset, output_folder, filename)

            time = time + time_step


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


class MRMSDataAnd(MRMSData):
    """
    Combines
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


    def process_day(
            self,
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(minutes=15),
            include_scan_time=False
    ):

        super().proces_day(
            domain,
            year,
            month,
            day,
            output_folder,
            path=None,
            time_step=timedelta(minutes=15),
            include_scan_time=False
        )




mrms_precip_rate = MRMSData(
    "mrms",
    4,
    [RetrievalTarget("surface_precip", 1e-3)],
    "rqi"
)

mrms_precip_rate_and_type = MRMSData(
    "mrms_w_type",
    4,
    [
        RetrievalTarget("surface_precip", 1e-3),
        RetrievalTarget("precip_type", None),
    ],
    "rqi"
)
