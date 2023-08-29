"""
cimr.areas
==========

Contains area definitions for the regions used by CIMR.
"""
from pathlib import Path

import numpy as np
import pyresample

from pansat.roi import ROI, PolygonROI

###############################################################################
# Nordics
###############################################################################

NORDICS_1 = pyresample.load_area(Path(__file__).parent / "cimr_nordic.yml")
NORDICS_2 = NORDICS_1[(slice(0, None, 2), slice(0, None, 2))]
NORDICS_4 = NORDICS_2[(slice(0, None, 2), slice(0, None, 2))]
NORDICS_8 = NORDICS_4[(slice(0, None, 2), slice(0, None, 2))]
NORDICS_16 = NORDICS_8[(slice(0, None, 2), slice(0, None, 2))]
ROI_NORDICS = ROI(
    -9.05380216185029,
    51.77251844681491,
    45.24074941367874,
    73.3321989854415
)
_lons, _lats = NORDICS_8.get_lonlats()
ROI_POLY_NORDICS =  PolygonROI(np.array([
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

CONUS_4 = pyresample.load_area(Path(__file__).parent / "cimr_conus.yml")
CONUS_8 = CONUS_4[(slice(0, None, 2), slice(0, None, 2))]
CONUS_16 = CONUS_8[(slice(0, None, 2), slice(0, None, 2))]
ROI_CONUS = ROI(
    -129.995,
    20.005,
    -60.005,
    54.995
)
_lons, _lats = CONUS_8.get_lonlats()
ROI_POLY_CONUS =  PolygonROI(np.array([
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
