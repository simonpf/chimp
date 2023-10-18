import numpy as np

from cimr.input_data.utils import (
    generate_input,
    scale_slices
)


def test_generate_input():

    #
    # 1D input
    #

    n_channels = 9
    size = (128,)
    rng = np.random.default_rng()

    x = generate_input(n_channels, size, "sparse", rng, None)
    assert x is None

    x = generate_input(n_channels, size, "random", rng, None)
    assert x is not None
    assert np.isclose(np.mean(x), 0, atol=1e-1)

    mean = np.arange(9)
    x = generate_input(n_channels, size, "mean", rng, mean)
    assert x is not None
    assert np.all(np.isclose(np.mean(x, -1), mean))

    #
    # 2D input
    #

    n_channels = 9
    size = (128, 128)
    rng = np.random.default_rng()

    x = generate_input(n_channels, size, "sparse", rng, None)
    assert x is None

    x = generate_input(n_channels, size, "random", rng, None)
    assert x is not None
    assert np.isclose(np.mean(x), 0, atol=1e-1)

    mean = np.arange(9)
    x = generate_input(n_channels, size, "mean", rng, mean)
    assert x is not None
    assert np.all(np.isclose(np.mean(x, (-2, -1)), mean))


def test_scale_slices():
    """
    Test scaling of slices works as expected.

    """
    slcs = scale_slices(None, 2)
    assert slcs is not None

    slcs = (slice(0, 2), slice(0, 2))
    slcs_scld = scale_slices(slcs, 2)
    assert slcs_scld[0].start == 0
    assert slcs_scld[0].stop == 1
    assert slcs_scld[1].start == 0
    assert slcs_scld[1].stop == 1

    slcs = (slice(0, 2), slice(0, 2))
    slcs_scld = scale_slices(slcs, 0.5)
    assert slcs_scld[0].start == 0
    assert slcs_scld[0].stop == 4
    assert slcs_scld[1].start == 0
    assert slcs_scld[1].stop == 4
