"""
Tests for the cimr.data.mhs module.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l1c_metopb_mhs
import pytest
import xarray as xr

from cimr.areas import NORDICS
from cimr.data.mhs import (
    process_file
)

DOMAIN = NORDICS
MHS_PRODUCT = l1c_metopb_mhs


@pytest.fixture(scope="session")
def mhs_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    start_time = datetime(2020, 1, 1)
    end_time = datetime(2020, 1, 2)
    provider = GesdiscProvider(MHS_PRODUCT)
    product_files = provider.get_files_in_range(start_time, end_time)

    for filename in product_files:
        swath = parse_swath(provider.download_metadata(filename))
        if swath.intersects(DOMAIN["roi_poly"].to_geometry()):
            break

    provider.download_file(filename, path / filename)
    return path / filename


def test_resampling(mhs_file):
    """
    Ensure that processing a single file produces a training data file
    with the expected input.
    """
    data_path = mhs_file.parent
    mhs_data = MHS_PRODUCT.open(mhs_file)
    process_file(
        NORDICS,
        mhs_data,
        mhs_file.parent,
        timedelta(minutes=15)
    )

    scenes = sorted(list(data_path.glob("*.nc")))
    assert len(scenes) > 0

    scene = xr.load_dataset(scenes[0])
    assert scene.channels.size == 5
