"""
cimr.areas
==========

Contains area definitions for the regions used by CIMR.
"""
from pathlib import Path

import pyresample
from pansat.roi import ROI

ROI_NORDIC = ROI(
    -9.05380216185029,
    51.77251844681491,
    45.24074941367874,
    73.3321989854415
)

NORDIC_1 = pyresample.load_area(Path(__file__).parent / "cimr_nordic.yml")
NORDIC_2 = NORDIC_1[(slice(0, None, 2), slice(0, None, 2))]
NORDIC_4 = NORDIC_2[(slice(0, None, 2), slice(0, None, 2))]
NORDIC_8 = NORDIC_4[(slice(0, None, 2), slice(0, None, 2))]


