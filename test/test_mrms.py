"""
Test processing of MRMS data.
"""
from datetime import timedelta
import os
from pathlib import Path

import pytest

import numpy as np
from pansat.products.ground_based import mrms
import xarray as xr

from cimr.data import mrms
from cimr import areas

data_path = Path(__file__).parent / "data"
precip_rate_file = "PrecipRate_00.00_20200101-103000.grib2.gz"
rqi_file = "RadarQualityIndex_00.00_20200101-103000.grib2.gz"

HAS_PANSAT = "PANSAT_PASSWORD" in os.environ
NEEDS_PANSAT = pytest.mark.skipif(not HAS_PANSAT, reason="pansat password not set.")

@pytest.fixture(scope="session")
def mrms_test_data(tmp_path_factor):

    destination = tmp_path_factory.mktemp("data")
    time_range = TimeRange("2020-01-01T00:00:00", "2020-01-01T00:02:00")

    precip_rate_file = mrms.precip_rate.find_files(time_range)[-1]
    precip_rate_file = precip_rate_file.download(destination=destination)
    rqi_file = mrms.rqi.find_file(time_range)[-1]
    rqi_file = rqi_file.download(destination=destination)
    precip_flag_file = mrms.precip_flag.find_file(time_range)[-1]
    precip_flag_file = precip_flag_file.download(destination=destination)

    precip_rate_data = mrms.precip_rate.open(precip_rate_file)
    rqi_data = mrms.radar_quality_index.open(rqi_file)
    precip_flag_file = mrms.precip_flag.open(precip_flag_file)

    dataset = xr.merge([precip_rate_data, rqi_data], compat="override").rename({
        "precip_rate": "surface_precip",
        "radar_quality_index": "rqi",
        "precip_flag": "precip_type"
    })
    return dataset

@NEEDS_PANSAT
def test_resampling(mrms_test_data):
    """
    Test resampling of MRMS data.
    """
    dataset_r = mrms.resample_mrms_data(mrms_test_data)
    assert np.any(np.isfinite(dataset_r.surface_precip.data))


def test_extract_data(tmp_path):
    "Test extraction of MRMS training data."
    time_step = timedelta(hours=24)
    mrms.process_day(areas.CONUS, 2020, 1, 1, tmp_path, time_step=time_step)

    files = list((tmp_path / "mrms").glob("*.nc"))
    assert len(files) > 0

    data = xr.load_dataset(files[0])
    assert "surface_precip" in data
    assert "rqi" in data
    assert "precip_type" in data
    assert np.any(np.isfinite(data.surface_precip))
