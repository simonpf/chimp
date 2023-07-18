"""
Tests for the cimr.metrics module.
"""
from scipy.fft import idctn
import numpy as np

from cimr.metrics import (
    Bias,
    MSE,
    Correlation,
    PRCurve,
    SpectralCoherence
)


def test_bias():
    """
    Test the calculation of the bias metric.
    """
    y_pred = {
        "surface_precip": np.zeros((32, 32)),
        "ice_water_path": np.zeros((32, 32)),
    }
    y_true = {
        "surface_precip": np.ones((32, 32)),
        "ice_water_path": np.ones((32, 32)),
    }

    bias = Bias()
    bias.calc(y_pred, y_true)
    bias.calc(y_pred, y_true)
    bias.calc(y_pred, y_true)

    results = bias.results()
    assert results["surface_precip_bias"] == -1.0
    assert results["surface_precip_rel_bias"] == -1.0
    assert results["ice_water_path_bias"] == -1.0
    assert results["ice_water_path_rel_bias"] == -1.0


def test_mse():
    """
    Test the calculation of the MSE metric.
    """
    y_pred = {
        "surface_precip": np.zeros((32, 32)),
        "ice_water_path": np.zeros((32, 32)),
    }
    y_true = {
        "surface_precip": np.ones((32, 32)),
        "ice_water_path": np.ones((32, 32)),
    }

    mse = MSE()
    mse.calc(y_pred, y_true)
    mse.calc(y_pred, y_true)
    mse.calc(y_pred, y_true)

    results = mse.results()
    assert results["surface_precip_mse"] == 1.0
    assert results["surface_precip_rel_mse"] == 1.0
    assert results["ice_water_path_mse"] == 1.0
    assert results["ice_water_path_rel_mse"] == 1.0


def test_corr():
    """
    Test the calculation of the correlation.
    """
    y_pred = {
        "surface_precip": np.random.rand(32, 32),
        "ice_water_path": np.random.rand(32, 32),
    }
    y_true = {
        "surface_precip": 2 * y_pred["surface_precip"],
        "ice_water_path": -2 * y_pred["ice_water_path"]
    }

    corr = Correlation()
    corr.calc(y_pred, y_true)
    corr.calc(y_pred, y_true)
    corr.calc(y_pred, y_true)

    results = corr.results()
    assert np.isclose(results["surface_precip_corr"].data, 1.0)
    assert np.isclose(results["ice_water_path_corr"].data, -1.0)


def test_corr():
    """
    Test the calculation of the correlation.
    """
    y_pred = {
        "surface_precip": np.linspace(0, 1, 101),
        "ice_water_path": np.linspace(0, 2, 101),
    }
    y_true = {
        "surface_precip": y_pred["surface_precip"] > 0.5,
        "ice_water_path": y_pred["ice_water_path"] > 0.5
    }

    pr_curve = PRCurve()
    pr_curve.calc(y_pred, y_true)
    pr_curve.calc(y_pred, y_true)
    pr_curve.calc(y_pred, y_true)

    results = pr_curve.results()

    prec = results["surface_precip_prec"].data
    rec = results["surface_precip_rec"].data

    assert np.isclose(np.nanmin(rec), 0.0)
    assert np.isclose(np.nanmax(rec), 1.0)
    assert np.isclose(np.nanmin(prec), 0.5)
    assert np.isclose(np.nanmax(prec), 1.0)


def _random_spectral_field(
        size,
        max_var,
        min_var
):
    """
    Create random spectral fields with similar specta up to a certain
    minimum scale of spatial variation.

    Args:
        size: Tuple ``(h, w)`` specifying the height and widt of the field.
        max_var: Maximum scale of spatial variability
        min_var: Minimum scale of spatial variability

    Return:
        A tuple ``(field, field_ret)`` containing a random fields ``field``,
        which exhibits spatial variability across all scales smaller than
        'max_var', and ``field_ret``, which exhibits spatial variability
        only at scales exceeding ``min_var``.
    """
    wny = 0.5 * np.arange(size[0]) / size[0]
    wnx = 0.5 * np.arange(size[1]) / size[1]
    wn = np.sqrt(wny ** 2 + wnx[..., None] ** 2)
    scale = 1 / wn

    coeffs = np.random.rand(*size)
    coeffs *= np.sqrt(scale)
    coeffs[:, 0] = 0.0
    coeffs[0, :] = 0.0
    coeffs[scale > max_var] = 0.0

    coeffs_ret = coeffs.copy()
    coeffs_ret[scale < min_var] = 0.0

    field = idctn(coeffs, norm="ortho")
    field_ret = idctn(coeffs_ret, norm="ortho")

    return field, field_ret


def test_spectral_coherence():
    """
    Test the calculation of the spectral coherence.
    """
    size = (2048, 2048)
    field, field_ret = _random_spectral_field(size, 32, 16)

    sc = SpectralCoherence(window_size=32, scale=1)
    sc.calc(field_ret, field)

    results = sc.results()

    er = results["effective_resolution"]
    assert np.isclose(er, 16.0, rtol=0.4)
