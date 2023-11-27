"""
Tests for the calculation of input data statistics.
"""
import numpy as np
import pytest
import xarray as xr

from chimp.bin.calculate_statistics import process_files
from chimp.data import get_input


@pytest.fixture
def input_files(tmp_path):

    data_dir = tmp_path / "files"
    data_dir.mkdir()

    files = []
    for ind in range(10):
        tbs = ind * np.ones((10, 20, 20))
        dataset = xr.Dataset({
            "tbs": (("channels", "y", "x"), tbs)
        })
        path = data_dir / f"{ind}.nc"
        dataset.to_netcdf(path)
        files.append(path)
    return files


def test_input_data_statistics(tmp_path, input_files):

    stats = process_files(get_input("GMI"), input_files, 2)
    stats.to_netcdf(tmp_path)
    stats = xr.load_dataset(tmp_path / "input_statistics.nc")

    print(stats)

    assert np.all(np.isclose(stats.tb_mean.data, 4.5))
    assert np.all(np.isclose(stats.tb_min.data, 0))
    assert np.all(np.isclose(stats.tb_max.data, 9))
