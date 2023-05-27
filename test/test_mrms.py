"""
Test processing of MRMS data.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
from pansat.products.ground_based.mrms import (
    mrms_precip_rate,
    mrms_radar_quality_index
)
import xarray as xr

from cimr.data import mrms
from cimr import areas

data_path = Path(__file__).parent / "data"
precip_rate_file = "PrecipRate_00.00_20200101-103000.grib2.gz"
rqi_file = "RadarQualityIndex_00.00_20200101-103000.grib2.gz"


def test_resampling():
    """
    Test resampling of MRMS data.
    """
    precip_rate_data = mrms_precip_rate.open(data_path / "mrms" / precip_rate_file)
    rqi_data = mrms_radar_quality_index.open(data_path / "mrms" / rqi_file)

    dataset = xr.merge([precip_rate_data, rqi_data]).rename({
        "precip_rate": "surface_precip",
        "radar_quality_index": "rqi"
    })
    dataset_r = mrms.resample_mrms_data(dataset)
    assert np.any(np.isfinite(dataset_r.surface_precip.data))


def test_extract_data(tmp_path):
    "Test extraction of MRMS training data."
    time_step = timedelta(hours=24)
    mrms.process_day(areas.CONUS, 2020, 1, 1, tmp_path, time_step=time_step)

    files = list((tmp_path / "radar").glob("*.nc"))
    assert len(files) > 0

    data = xr.load_dataset(files[0])
    assert np.any(np.isfinite(data.surface_precip))
