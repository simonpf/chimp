"""
===========
chimp.areas
===========

Contains area definitions for the regions used by CHIMP.
"""
from pathlib import Path

import numpy as np
import pyresample

from pansat.geometry import Polygon, LonLatRect

###############################################################################
# Nordics
###############################################################################

NORDICS_1 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_1.yml")
NORDICS_2 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_2.yml")
NORDICS_4 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_4.yml")
NORDICS_8 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_8.yml")
NORDICS_16 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_16.yml")
ROI_NORDICS = LonLatRect(
    -9.05380216185029,
    51.77251844681491,
    45.24074941367874,
    73.3321989854415
)
_lons, _lats = NORDICS_8.get_lonlats()
ROI_POLY_NORDICS =  Polygon(np.array([
    [_lons[0, 0], _lats[0, 0]],
    [_lons[0, -1], _lats[0, -1]],
    [_lons[-1, -1], _lats[-1, -1]],
    [_lons[-1, 0], _lats[-1, 0]],
]))

NORDICS = {
    1: NORDICS_1,
    2: NORDICS_2,
    4: NORDICS_4,
    8: NORDICS_8,
    16: NORDICS_16,
    "roi": ROI_NORDICS,
    "roi_poly": ROI_POLY_NORDICS
}

###############################################################################
# CONUS
###############################################################################

CONUS_4 = pyresample.load_area(Path(__file__).parent / "chimp_conus_4.yml")
CONUS_8 = pyresample.load_area(Path(__file__).parent / "chimp_conus_8.yml")
CONUS_16 = pyresample.load_area(Path(__file__).parent / "chimp_conus_16.yml")
ROI_CONUS = LonLatRect(
    -129.995,
    20.005, -60.005,
    54.995
)
_lons, _lats = CONUS_8.get_lonlats()
ROI_POLY_CONUS =  Polygon(np.array([
    [_lons[0, 0], _lats[0, 0]],
    [_lons[0, -1], _lats[0, -1]],
    [_lons[-1, -1], _lats[-1, -1]],
    [_lons[-1, 0], _lats[-1, 0]],
]))

CONUS = {
    4: CONUS_4,
    8: CONUS_8,
    16: CONUS_16,
    "roi": ROI_CONUS,
    "roi_poly": ROI_POLY_CONUS
}

MRMS = pyresample.load_area(Path(__file__).parent / "mrms.yml")

###############################################################################
# EUROPE
###############################################################################

EUROPE_4 = pyresample.load_area(Path(__file__).parent / "chimp_conus_4.yml")
EUROPE_8 = pyresample.load_area(Path(__file__).parent / "chimp_conus_8.yml")
EUROPE_16 = pyresample.load_area(Path(__file__).parent / "chimp_conus_16.yml")
ROI_EUROPE = LonLatRect(
    -129.995,
    20.005, -60.005,
    54.995
)
_lons, _lats = EUROPE_8.get_lonlats()
ROI_POLY_EUROPE =  Polygon(np.array([
    [_lons[0, 0], _lats[0, 0]],
    [_lons[0, -1], _lats[0, -1]],
    [_lons[-1, -1], _lats[-1, -1]],
    [_lons[-1, 0], _lats[-1, 0]],
]))

EUROPE = {
    4: EUROPE_4,
    8: EUROPE_8,
    16: EUROPE_16,
    "roi": ROI_EUROPE,
    "roi_poly": ROI_POLY_EUROPE
}

MERRA = pyresample.load_area(Path(__file__).parent / "merra.yml")
###############################################################################
# MERRA
###############################################################################

MERRA = pyresample.load_area(Path(__file__).parent / "merra.yml")
