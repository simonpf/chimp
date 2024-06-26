"""
Tests for the chimp.areas module.
"""
import pyresample

import numpy as np

from chimp import areas


def test_get_area():
    """
    Ensure accessing area definitions works for single-scale areas.
    """
    nordics = areas.get_area("nordics")
    assert id(nordics) == id(areas.NORDICS)


def test_getitem():
    """
    Ensure accessing area definitions works for single-scale areas.
    """
    nordics_2 = areas.NORDICS[2]
    assert isinstance(nordics_2, pyresample.AreaDefinition)

    merra = areas.MERRA[2]
    assert isinstance(merra, pyresample.AreaDefinition)


def test_global():
    """
    Ensure that the global latlon grid has a resolution of ~ 0.05 degree.
    """
    glbl = areas.get_area("global_latlon")

    scale = 4
    for _ in range(3):
        glbl_x = glbl[scale]
        lons, lats = glbl_x.get_lonlats()
        assert np.isclose(np.diff(lons[0]).max(), 0.05 * (scale // 4))
        assert np.isclose(np.diff(lats[..., 0]).max(), -0.05 * (scale // 4))
        scale *= 2
