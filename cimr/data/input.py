"""
cimr.data.input
===============

This sub-module defines classes for representing different input types.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
from quantnn.normalizer import Normalizer, MinMaxNormalizer
from scipy import ndimage
import xarray as xr

from cimr.data.utils import scale_slices, generate_input


def find_random_scene(
        path,
        rng,
        multiple=4,
        window_size=256,
        rqi_thresh=0.8,
        valid_fraction=0.2
):
    """
    Finds a random crop in the input data that is guaranteeed to have
    valid observations.

    Args:
        path: The path of the reference data file.
        rng: A numpy random generator instance to randomize the scene search.
        multiple: Limits the scene position to coordinates that a multiples
            of the given value.
        rqi_thresh: Threshold for the minimum RQI of a reference pixel to
            be considered valid.
        valid_fraction: The minimum amount of valid samples in the
            region.

    Return:
        A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
        of the random crop.
    """
    with xr.open_dataset(path) as data:

        if "latitude" in data.dims:
            n_rows = data.latitude.size
            n_cols = data.longitude.size
        else:
            n_rows = data.y.size
            n_cols = data.x.size

        if "swath_centers" in data.dims:

            row_inds = data.swath_center_row_inds.data
            col_inds = data.swath_center_col_inds.data

            valid = (
                (row_inds > window_size // 2) *
                (row_inds < n_rows - window_size // 2) *
                (col_inds > window_size // 2) *
                (col_inds < n_cols - window_size // 2)
            )

            if valid.sum() == 0:
                return None


            row_inds = row_inds[valid]
            col_inds = col_inds[valid]

            ind = rng.choice(np.arange(valid.sum()))
            row_c = row_inds[ind]
            col_c = col_inds[ind]

            i_start = (row_c - window_size // 2) // multiple * multiple
            i_end = i_start + window_size
            j_start = (col_c - window_size // 2) // multiple * multiple
            j_end = j_start + window_size

        else:
            i_start = rng.integers(0, (n_rows - window_size) // multiple)
            i_end = i_start + window_size // multiple
            j_start = rng.integers(0, (n_cols - window_size) // multiple)
            j_end = j_start + window_size // multiple

            i_start = i_start * multiple
            i_end = i_end * multiple
            j_start = j_start * multiple
            j_end = j_end * multiple

    return (i_start, i_end, j_start, j_end)




class InputBase:
    """
    Base class for all inputs that keeps track of all instances.
    """
    ALL_INPUTS = {}

    def __init__(self, name):
        self.ALL_INPUTS[name] = self

    @classmethod
    def get_input(cls, name: str) -> "InputBase":
        """
        Get an input object by its name.

        Return:
            The input object of the given name if it exists.

        Raises:
            ValueError if there is no input with the given name
        """
        if not name in cls.ALL_INPUTS:
            raise ValueError(
                f"Input '{name}' is not a known input. Currently known inputs "
                f" are: {list(cls.ALL_INPUTS.keys())}"
            )
        return cls.ALL_INPUTS[name]


def get_input(name: Union[str, InputBase]) -> InputBase:
    """
    Get an input object by its name.

    Raises:
        ValueError if there is no input with the given name
    """
    if isinstance(name, InputBase):
        return name
    return InputBase.get_input(name)


def get_inputs(input_list: List[Union[str, InputBase]]) -> List[InputBase]:
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


class MinMaxNormalized:
    """
    Mixin' class provides a quantnn.normalizer.MinMaxNormalizer with
    statistics read from a file.
    """
    def __init__(self, stats_file):
        self._stats_file = stats_file
        self._normalizer = None

    @property
    def normalizer(self):
        """
        Cached access to normalizer.

        On first access this will trigger a read of the corresponding
        stats file.
        """
        if self._normalizer is None:
            path = Path(__file__).parent / "stats"
            stats_file = path / f"{self._stats_file}.txt"
            if not stats_file.exists():
                raise RuntimeError(
                    f"Could not find the stats file {stats_file}."
                )
            stats = np.loadtxt(stats_file, skiprows=1).reshape(-1, 2)
            norm = MinMaxNormalizer(
                np.ones((stats.shape[0], 1, 1)),
                feature_axis=0
            )
            for chan_ind in range(stats.shape[0]):
                norm.stats[chan_ind] = tuple(stats[chan_ind])
            self._normalizer = norm
        return self._normalizer


@dataclass
class Input(InputBase, MinMaxNormalized):
    """
    Record holding the paths of the files for a single training
    sample.
    """
    name: str
    scale: int
    variables: Union[str, List[str]]
    mean: Optional[np.array] = None
    n_dim: int = 2

    def __init__(
            self,
            name: str,
            scale: int,
            variables: Union[str, List[str]],
            mean: Optional[np.array] = None,
            n_dim: int = 2
    ):
        InputBase.__init__(self, name)
        MinMaxNormalized.__init__(self, name)

        self.name = name
        self.scale = scale
        self.variables = variables
        self.mean = mean
        self.n_dim = n_dim

    @property
    def n_channels(self):
        return len(self.normalizer.stats)

    def load_sample(
            self,
            input_file: Path,
            crop_size: Union[int, Tuple[int, int]],
            base_scale: int,
            slices: Tuple[slice, slice],
            rng: np.random.Generator,
            missing_input_policy: str,
            rotate: Optional[float] = None,
            flip: Optional[bool] = False,
            normalize: Optional[bool] = True
    ) -> np.ndarray:
        """
        Load input data sample from file.

        Args:
            input_file: Path pointing to the input file from which to load the
                data.
            crop_size: Size of the final crop.
            base_scale: The scale of the reference data.
            sclices: Tuple of slices defining the part of the data to load.
            rng: A numpy random generator object to use to generate random
                data.
            missing_input_policy: A string describing how to handle missing
                input.
            rotate: If given, the should specify the number of degree by
                which the input should be rotated.
            flip: Bool indicated whether or not to flip the input along the
                last dimensions.
            normalizer: Whether or not the inputs should be normalized.
        """
        rel_scale = base_scale / self.scale

        if isinstance(crop_size, int):
            crop_size = (crop_size,) * self.n_dims
        crop_size = tuple((int(size / rel_scale) for size in crop_size))

        if input_file is None:
            x_s = generate_input(
                self.n_channels,
                crop_size,
                missing_input_policy,
                rng,
                self.mean
            )
            # TODO: Handle normalization more elegantly.
            if x_s is not None and normalize:
                x_s = self.normalizer(x_s)
            return x_s

        row_slice, col_slice = scale_slices(slices, rel_scale)


        with xr.open_dataset(input_file) as data:
            vars = self.variables
            if isinstance(vars, str):
                x_s = data[vars][..., row_slice, col_slice].data
                # Expand dims in case of single-channel inputs.
                if x_s.ndim < 3:
                    x_s = x_s[None]
            else:
                x_s = np.stack(
                    [data[vrbl][row_slice, col_slice].data for vrbl in vars]
                )

            # Apply augmentations
            if rotate is not None:
                x_s = ndimage.rotate(
                    x_s,
                    rotate,
                    order=0,
                    reshape=False,
                    axes=(-2, -1),
                    cval=np.nan
                )
                height = x_s.shape[-2]

                # In case of a rotation, we may need to cut off some input.
                height_out = int(crop_size[0] / rel_scale)
                if height > height_out:
                    start = (height - height_out) // 2
                    end = start + height_out
                    x_s = x_s[..., start:end, :]

                width = x_s.shape[-1]
                width_out = int(crop_size[1] / rel_scale)
                if width > width_out:
                    start = (width - width_out) // 2
                    end = start + width_out
                    x_s = x_s[..., start:end]
            if flip:
                x_s = np.flip(x_s, -1)

        if normalize:
            x_s = self.normalizer(x_s)

        return x_s


GOES = Input("goes", 4, [f"geo_{ind:02}" for ind in range(1, 12)])
