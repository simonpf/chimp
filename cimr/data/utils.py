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
from cimr.data import inputs, reference


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


def get_input(inpt: Union[str, inputs.Input]) -> inputs.Input:
    """
    Parse input object.

    For simplicity retrieval inputs can be specified as strings
    or 'cimr.data.inputs.Input' object. This function replaces
    traverses the given list of inputs and replaces strings with
    the corresponding predefined 'Input' object.

    Args:
        input_list: List containing strings of 'cimr.data.inputs.Input'
            objects.

    Return:
        A new list containing only 'cimr.data.inputs.Input' objects.
    """
    if isinstance(inpt, inputs.Input):
        return inpt

    try:
        inpt_obj = getattr(inputs, inpt.upper())
    except AttributeError:
        raise ValueError(f"The input '{inpt}' is not known.")
    return inpt_obj


def get_inputs(input_list: List[Union[str, inputs.Input]]) -> List[inputs.Input]:
    """
    Parse input object.

    For simplicity retrieval inputs can be specified as strings
    or 'cimr.data.inputs.Input' object. This function replaces
    traverses the given list of inputs and replaces strings with
    the corresponding predefined 'Input' object.

    Args:
        input_list: List containing strings of 'cimr.data.inputs.Input'
            objects.

    Return:
        A new list containing only 'cimr.data.inputs.Input' objects.
    """
    return [get_input(inpt) for inpt in input_list]


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
