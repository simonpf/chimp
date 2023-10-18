"""
cimr.data.input.utils
=====================

Utility functions for the handling of input data.
"""
from typing import Optional, Tuple, Union

import numpy as np

def generate_input(
        n_channels,
        size: Tuple[int],
        policy: str,
        rng: np.random.Generator,
        mean: Optional[np.array],
):
    """
    Generate input values for missing inputs.

    Args:
        n_channels: The number of channels in the input.
        size: Tuple defining the spatial dimensions of the input to
            generate.
        policy: The policy to use for the data generation.
        rng: The random generator object to use to create random
            arrays.
        mean: An array containing 'n_channels' defining the mean of
            each channel.

    Return:
        An numpy.ndarray containing replacement data.
    """
    if policy == "sparse":
        return None

    elif policy == "random":
        return rng.normal(size=(n_channels,) + size)
    elif policy == "mean":
        if mean is None:
            raise RuntimeError(
                "If missing-input policy is 'mean', an array of mean values"
                " must be provided."
            )

        return mean[(slice(0, None),) + (None,) * len(size)] * np.ones(
            shape=(n_channels,) + size,
            dtype="float32"
        )
    elif policy == "missing":
        return np.nan * np.ones(
            shpare=(n_channels,) + size,
            dtype="float32"
        )

    raise ValueError(
        f"Missing input policy '{policy}' is not known. Choose between 'sparse'"
        " 'random', 'mean' and 'missing'. "
    )


def scale_slices(
        slices: Union[Tuple[slice, slice], None],
        rel_scale: float
) -> Tuple[slice, slice]:
    """
    Scale slices to match input scale.

    Args:
        slices: A tuple containing row- and column-slices defining the
            input region to extract with respect to the reference data. If
            'None', slices corresponding to the full
        scale: The scale of the input data relative to the reference data.

    Return:
        A tuple contining the slices scaled to match the relative scale of
        the input data.
    """
    if slices is None:
        return (slice(0, None), slice(0, None))

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
