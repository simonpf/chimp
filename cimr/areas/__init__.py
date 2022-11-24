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

NORDIC_1 = pyresample.load_area(Path(__file__).parent / "cimr_nordic.yml")
NORDIC_2 = NORDIC_1[(slice(0, None, 2), slice(0, None, 2))]
NORDIC_4 = NORDIC_2[(slice(0, None, 2), slice(0, None, 2))]
NORDIC_8 = NORDIC_4[(slice(0, None, 2), slice(0, None, 2))]
NORDIC_16 = NORDIC_8[(slice(0, None, 2), slice(0, None, 2))]
ROI_NORDIC = ROI(
    -9.05380216185029,
    51.77251844681491,
    45.24074941367874,
    73.3321989854415
)
_lons, _lats = NORDIC_8.get_lonlats()
ROI_POLY_NORDIC =  PolygonROI(np.array([
    [_lons[0, 0], _lats[0, 0]],
    [_lons[0, -1], _lats[0, -1]],
    [_lons[-1, -1], _lats[-1, -1]],
    [_lons[-1, 0], _lats[-1, 0]],
]))

NORDIC = {
    1: NORDIC_1,
    2: NORDIC_2,
    4: NORDIC_4,
    8: NORDIC_8,
    16: NORDIC_16,
    "roi": ROI_NORDIC,
    "roi_poly": ROI_POLY_NORDIC
}



