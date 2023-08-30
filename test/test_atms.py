"""
Tests for the cimr.data.atms module.
"""
from datetime import datetime, timedelta
from pathlib import Path

from conftest import NEEDS_PANSAT
import numpy as np
import pytest
import xarray as xr

from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l1c_noaa20_atms

from cimr.areas import NORDICS
from cimr.data.atms import (
    process_file
)


DOMAIN = NORDICS
ATMS_PRODUCT = l1c_noaa20_atms


@pytest.fixture
def atms_file(tmp_path):
    start_time = datetime(2020, 1, 1)
    end_time = datetime(2020, 1, 2)
    provider = GesdiscProvider(ATMS_PRODUCT)
    product_files = provider.get_files_in_range(start_time, end_time)

    for filename in product_files:
        swath = parse_swath(provider.download_metadata(filename))
        if swath.intersects(DOMAIN["roi_poly"].to_geometry()):
            break

    provider.download_file(filename, tmp_path / filename)
    return tmp_path / filename


@pytest.mark.slow
@NEEDS_PANSAT
def test_process_file(atms_file):
    """
    Enusre that processing a single file produces a training data file
    with the expected input.
    """
    data_path = atms_file.parent
    atms_data = ATMS_PRODUCT.open(atms_file)
    process_file(
        NORDICS,
        atms_data,
        atms_file.parent,
        timedelta(minutes=15)
    )

    scenes = sorted(list(data_path.glob("*.nc")))
    assert len(scenes) > 0

    scene = xr.load_dataset(scenes[0])
    assert scene.channels.size == 9
