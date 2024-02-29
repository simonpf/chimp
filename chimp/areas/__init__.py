"""
===========
chimp.areas
===========

This module defines all the areas, i.e., regional domains over which data is extracted
used for generating CHIMP training datasets.
"""
from typing import Dict, Union
from pathlib import Path

import numpy as np
import pyresample

from pansat.geometry import Geometry, Polygon, LonLatRect, lonlats_to_polygon


ALL_AREAS = {}


class Area:
    """
    The Area class represents the different spatial domains over which CHIMP training
    data can be extracted. An area can support multi-scale and single-scale retrievals.
    For single scale retrievals, the area object holds a single area definition and all
    input and reference data is mapped to the definition. When an area holds a dict of
    area definitions, input and reference data is extract at their corresponding native
    distributions.
    """
    def __init__(
            self,
            name: str,
            areas: Union[pyresample.AreaDefinition, Dict[int, pyresample.AreaDefinition]],
    ):
        """
        Instantiate a CHIMP area.

        Args:
            name: The name that uniquely identifies the area.
            area: A dict mapping scales to corresponding area definitions.
        """
        self.name = name
        self.areas = areas
        ALL_AREAS[name.lower()] = self
        lons, lats = self[8].get_lonlats()
        self.roi = lonlats_to_polygon(
            lons, lats, 6
        )

    def __getitem__(self, scale: int) -> pyresample.AreaDefinition:
        """
        Access area definition for a given scale.

        Args:
            scale: The scale for which to retrieve the area definition.
        """
        if isinstance(self.areas, pyresample.AreaDefinition):
            return self.areas
        return self.areas[scale]


def get_area(name: str) -> Area:
    """
    Retrieve area by its name.

    Args:
        name: The name of the area.

    Return:
        The area registered in the given name.

    Raises:
        Runtime error if the area name is not defined.
    """
    name = name.lower()
    area = ALL_AREAS.get(name, None)
    if area is None:
        raise RuntimeError(
            f"The area '{area}' isn't currently defined. Defined areas are '{list(ALL_AREAS.keys())}'."
        )
    return area




###############################################################################
# Nordics
###############################################################################

NORDICS_1 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_1.yml")
NORDICS_2 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_2.yml")
NORDICS_4 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_4.yml")
NORDICS_8 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_8.yml")
NORDICS_16 = pyresample.load_area(Path(__file__).parent / "chimp_nordic_16.yml")

NORDICS = Area(
    name="nordics",
    areas={
        1: NORDICS_1,
        2: NORDICS_2,
        4: NORDICS_4,
        8: NORDICS_8,
        16: NORDICS_16,
    }
)

###############################################################################
# CONUS
###############################################################################

CONUS_4 = pyresample.load_area(Path(__file__).parent / "chimp_conus_4.yml")
CONUS_8 = pyresample.load_area(Path(__file__).parent / "chimp_conus_8.yml")
CONUS_16 = pyresample.load_area(Path(__file__).parent / "chimp_conus_16.yml")


CONUS = Area(
    name="conus",
    areas={
        4: CONUS_4,
        8: CONUS_8,
        16: CONUS_16,
    },
)

MRMS = Area(
    "mrms",
    areas=pyresample.load_area(Path(__file__).parent / "mrms.yml")
)

CONUS_PLUS_4 = pyresample.load_area(Path(__file__).parent / "chimp_conus_plus_4.yml")
CONUS_PLUS_8 = pyresample.load_area(Path(__file__).parent / "chimp_conus_plus_8.yml")
CONUS_PLUS_16 = pyresample.load_area(Path(__file__).parent / "chimp_conus_plus_16.yml")

CONUS_PLUS = Area(
    name="conus_plus",
    areas={
        4: CONUS_PLUS_4,
        8: CONUS_PLUS_8,
        16: CONUS_PLUS_16,
    },
)


###############################################################################
# EUROPE
###############################################################################

EUROPE_4 = pyresample.load_area(Path(__file__).parent / "chimp_europe_4.yml")
EUROPE_8 = pyresample.load_area(Path(__file__).parent / "chimp_europe_8.yml")
EUROPE_16 = pyresample.load_area(Path(__file__).parent / "chimp_europe_16.yml")

EUROPE = Area(
    name="europe",
    areas={
        4: EUROPE_4,
        8: EUROPE_8,
        16: EUROPE_16,
    }
)

###############################################################################
# MERRA
###############################################################################

MERRA = Area(
    "merra",
    areas=pyresample.load_area(Path(__file__).parent / "merra.yml")
)
