"""
cimr.data.utils
===============

Utility functions used by the sub-modules of the cimr.data module.
"""
from pathlib import Path
from typing import Union, List

import numpy as np
import xarray as xr

from cimr.definitions import N_CHANS
from cimr.data import reference


def make_microwave_band_array(domain, band):
    """
    Create an empty array to hold  microwave observation data over the
    given domain.

    Args:
        domain: A domain dict describing the domain for which training data
            is created.
        band: The band name, i.e. one of ['mw_low', 'mw_90', 'mw_160', 'mw_183']

    Return:
        An array filled with nan's with the shape of the observation data
        for the given domain.
    """
    n_chans = N_CHANS[band]
    if band == "mw_low":
        shape = domain[16].shape
    else:
        shape = domain[8].shape
    shape = shape + (n_chans,)

    return xr.DataArray(np.nan * np.ones(shape, dtype=np.float32))


def get_reference_data(ref):
    """
    Parse reference dataset.

    The reference datasets are handled in a similar manner as the inputs.

    Args:
        ref: Either a 'cimr.data.reference.ReferenceData' object or
            the name of an attribute of the 'cimr.data.reference' module
            pointing to such an object.

    Return:
        A 'cimr.data.reference.ReferenceData' object representing the
        requested reference data.
    """
    if isinstance(ref, reference.ReferenceData):
        return ref

    try:
        return getattr(reference, ref.upper())
    except AttributeError:
        raise ValueError(f"The reference data '{ref}' is not known.")
