from pathlib import Path
import os

import numpy as np
import pytest

from cimr.tiling import parse_shape, Tiler
from cimr.data.training_data import CIMRPretrainDataset


TEST_DATA = os.environ.get("CIMR_TEST_DATA", None)
if TEST_DATA is not None:
    TEST_DATA = Path(TEST_DATA)
NEEDS_TEST_DATA = pytest.mark.skipif(
    TEST_DATA is None, reason="Needs 'CCIC_TEST_DATA'."
)

def test_parse_shape():
    """
    Test shape-parsing for different input types.
    """
    x = [np.zeros((512, 512)), np.zeros((256, 256))]
    shape = parse_shape(x)
    assert shape == (512, 512)

    x = {
        "x": np.zeros((512, 512)),
        "y": np.zeros((256, 256))
    }
    shape = parse_shape(x)
    assert shape == (512, 512)

    x = parse_shape(np.zeros((512, 512)))
    assert shape == (512, 512)


@NEEDS_TEST_DATA
def test_sparse_data():
    """
    Test that missing inputs are set to None.
    """
    training_data = CIMRPretrainDataset(
        TEST_DATA / "training_data",
        reference_data="mrms",
        inputs=["cpcir", "gmi"],
        sparse=True,
        window_size=128
    )
    x, y = training_data[7]

    tiler = Tiler((x, y), tile_size=38, overlap=8)

    x_t, y_t = tiler.get_tile(0, 0)

    assert x_t["cpcir"].shape[-2:] == (38, 38)
    assert x_t["gmi"].shape[-2:] == (19, 19)


def test_predict():
    """
    Ensures that the 'predict' function of the tiler correctly assembles
    results.
    """
    size = tuple(map(lambda x: 2 * x, np.random.randint(10, 20, size=2)))
    tile_size = tuple(map(lambda x: 2 * x, np.random.randint(3, 5, size=2)))
    overlap = np.random.randint(0, 1) * 2

    y = np.arange(size[0]).astype(np.float32)
    x = np.arange(size[1]).astype(np.float32)
    x = np.stack(np.meshgrid(x, y))

    def predict_fun(x):
        return {"x": x, "x_2": x[..., ::2, ::2]}

    tiler = Tiler(x, tile_size=tile_size, overlap=overlap)
    results = tiler.predict(predict_fun)

    x_p = results["x"]
    assert np.all(x_p == x)

    x_p = results["x_2"]
    assert np.all(x_p == x[..., ::2, ::2])
