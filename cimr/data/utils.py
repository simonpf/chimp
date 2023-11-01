"""
cimr.data.utils
===============

Utility functions used by the sub-modules of the cimr.data module.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from quantnn.normalizer import Normalizer

import torch

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


def generate_input(
        n_channels,
        size: Tuple[int],
        policy: str,
        rng: np.random.Generator,
        normalizer: Optional[Normalizer] = None,
        mean: Optional[np.array] = None,
) -> torch.Tensor:
    """
    Generate input values for missing inputs.

    Args:
        n_channels: The number of channels in the input.
        size: Tuple defining the spatial dimensions of the input to
            generate.
        policy: The policy to use for the data generation.
        rng: The random generator object to use to create random
            arrays.
        normalizer: If provided, will be used to normalize inputs that
            should be normalized.
        mean: An array containing 'n_channels' defining the mean of
            each channel.

    Return:
        A torch.Tesnro containing the generated input.
    """
    if policy == "sparse":
        return None

    elif policy == "random":
        tensor = rng.normal(size=(n_channels,) + size).astype(np.float32)
        return torch.tensor(tensor)
    elif policy == "mean":
        if mean is None:
            raise RuntimeError(
                "If missing-input policy is 'mean', an array of mean values"
                " must be provided."
            )
        tensor = mean[(slice(0, None),) + (None,) * len(size)] * np.ones(
            shape=(n_channels,) + size,
            dtype="float32"
        )
        if normalizer is not None:
            tensor = normalizer(tensor)
        return torch.tensor(tensor)
    elif policy == "missing":
        tensor = np.nan * np.ones(
            shape=(n_channels,) + size,
            dtype="float32"
        )
        if normalizer is not None:
            tensor = normalizer(tensor)
        return tensor

    raise ValueError(
        f"Missing input policy '{policy}' is not known. Choose between 'sparse'"
        " 'random', 'mean' and 'missing'. "
    )


def scale_slices(
        slices: Union[Tuple[slice, slice], Tuple[int, int, int, int], None],
        rel_scale: float
) -> Tuple[slice, slice]:
    """
    Scale slices to match input scale.

    Args:
        slices: A tuple containing row- and column-slices defining the
            input region to extract with respect to the reference data.
            The tuple can also be of length 4 and contain the start and
            stop values for the rows followed by those for the columns.
            If 'None', slices corresponding to the full row and column
            extent will be returned.
        scale: The scale of the input data relative to the reference data.

    Return:
        A tuple contining the slices scaled to match the relative scale of
        the input data.
    """
    if slices is None:
        return (slice(0, None), slice(0, None))

    if len(slices) == 4:
        slices = (
            slice(slices[0], slices[1]),
            slice(slices[2], slices[3])
        )

    if rel_scale == 1:
        return slices

    row_start = slices[0].start
    row_end = slices[0].stop
    row_step = slices[0].step
    col_start = slices[1].start
    col_end = slices[1].stop
    col_step = slices[1].step
    row_slice = slice(
        int(row_start / rel_scale),
        int(row_end / rel_scale),
        row_step
    )
    col_slice = slice(
        int(col_start / rel_scale),
        int(col_end / rel_scale),
        col_step
    )
    return (row_slice, col_slice)
