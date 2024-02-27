"""
Tests for the chimp.areas module.
"""
import pyresample

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
