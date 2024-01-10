"""
chimp.data.input
===============

This sub-module defines classes for representing different input types.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
from quantnn.normalizer import Normalizer, MinMaxNormalizer
from quantnn.masked_tensor import MaskedTensor
from scipy import ndimage
import torch
import xarray as xr

from chimp.data.utils import scale_slices, generate_input
from chimp.data.source import DataSource


def find_random_scene(
    path, rng, multiple=4, window_size=256, rqi_thresh=0.8, valid_fraction=0.2
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
                (row_inds > window_size // 2)
                * (row_inds < n_rows - window_size // 2)
                * (col_inds > window_size // 2)
                * (col_inds < n_cols - window_size // 2)
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


class InputBase(DataSource):
    """
    Base class for all inputs that keeps track of all instances.
    """

    ALL_INPUTS = {}

    def __init__(self, name):
        super().__init__(name)
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
        name = name.lower()
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
    from . import seviri
    from . import gpm

    if isinstance(name, InputBase):
        return name
    return InputBase.get_input(name)


def get_inputs(input_list: List[Union[str, InputBase]]) -> List[InputBase]:
    """
    Parse input object.

    For simplicity retrieval inputs can be specified as strings
    or 'chimp.data.inputs.Input' object. This function replaces
    traverses the given list of inputs and replaces strings with
    the corresponding predefined 'Input' object.

    Args:
        input_list: List containing strings of 'chimp.data.inputs.Input'
            objects.

    Return:
        A new list containing only 'chimp.data.inputs.Input' objects.
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
                raise RuntimeError(f"Could not find the stats file {stats_file}.")
            stats = np.loadtxt(stats_file, skiprows=1).reshape(-1, 2)
            norm = MinMaxNormalizer(
                np.ones((stats.shape[0], 1, 1)), feature_axis=0, replace_nan=False
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
        n_dim: int = 2,
        spatial_dims: Tuple[str, str] = ("y", "x"),
    ):
        InputBase.__init__(self, name)
        MinMaxNormalized.__init__(self, name)

        self.name = name
        self.scale = scale
        self.variables = variables
        self.mean = mean
        self.n_dim = n_dim
        self.spatial_dims = spatial_dims[: self.n_dim]

    @property
    def n_channels(self):
        return len(self.normalizer.stats)

    def replace_missing(self, tensor, missing_value_policy, rng):
        """
        Replace missing values in tensor.

        Args:
            tensor: A torch.tensor containing the input data for a
                single sample.
            missing_value_policy: Policy describing how to replace missing
                values.
            rng: A numpy random generator object.

        Return:
            A new tensor with NAN's replaced according to the missing value
            policy.
        """
        mask = torch.isnan(tensor)
        if missing_value_policy == "random":
            repl = rng.normal(size=tensor.shape).astype(np.float32)
            tensor = torch.where(mask, torch.tensor(repl), tensor)
        elif missing_value_policy == "mean":
            if self.mean is None:
                raise RuntimeError(
                    "If missing-value policy is 'mean', an array of mean values"
                    " must be provided."
                )
            means = self.mean[(slice(0, None),) + (None,) * self.n_dim] * np.ones(
                shape=(self.n_channels,) + (1,) * self.n_dim, dtype="float32"
            )
            means = self.normalizer(means)
            tensor = torch.where(mask, torch.tensor(means), tensor)
        elif missing_value_policy == "missing":
            tensor = torch.where(mask, -1.5, tensor)
        elif missing_value_policy == "masked":
            tensor = MaskedTensor(tensor.to(dtype=torch.float32), mask=mask)
        elif missing_value_policy == "none":
            pass
        else:
            raise ValueError(
                f"Missing input policy '{missing_value_policy}' is not known. Choose between 'sparse'"
                " 'random', 'mean' and 'missing'. "
            )
        return tensor

    def find_files(self, base_path: Path) -> List[Path]:
        """
        Find input files.

        Args:
            base_path: Base path of the data containing the input data in a
                sub-folder.

        Return:
            List of available input files.
        """
        pattern = "*????????_??_??.nc"
        return sorted(list((base_path / self.name).glob(pattern)))

    def load_sample(
        self,
        input_file: Path,
        crop_size: Union[int, Tuple[int, int]],
        base_scale: int,
        slices: Tuple[slice, slice],
        rng: np.random.Generator,
        missing_value_policy: str,
        rotate: Optional[float] = None,
        flip: Optional[bool] = False,
    ) -> torch.Tensor:
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
            missing_value_policy: A string describing how to handle missing
                values.
            rotate: If given, the should specify the number of degree by
                which the input should be rotated.
            flip: Bool indicated whether or not to flip the input along the
                last dimensions.

        Return:
            A torch tensor containing the loaded input data.
        """
        rel_scale = self.scale / base_scale

        if isinstance(crop_size, int):
            crop_size = (crop_size,) * self.n_dims
        crop_size = tuple((int(size / rel_scale) for size in crop_size))

        if input_file is not None:
            slices = scale_slices(slices, rel_scale)
            with xr.open_dataset(input_file) as data:
                vars = self.variables
                if not isinstance(vars, list):
                    vars = [vars]
                all_data = []
                for vrbl in vars:
                    x_s = data[vrbl][dict(zip(self.spatial_dims, slices))].data
                    if x_s.ndim < 3:
                        x_s = x_s[None]
                    x_s = np.transpose(x_s, (2, 0, 1))
                    all_data.append(x_s)
                x_s = np.concatenate(all_data, axis=0)

            # Apply augmentations
            if rotate is not None:
                x_s = ndimage.rotate(
                    x_s, rotate, order=0, reshape=False, axes=(-2, -1), cval=np.nan
                )
                height = x_s.shape[-2]

                # In case of a rotation, we may need to cut off some input.
                height_out = crop_size[0]
                if height > height_out:
                    start = (height - height_out) // 2
                    end = start + height_out
                    x_s = x_s[..., start:end, :]

                width = x_s.shape[-1]
                width_out = crop_size[1]
                if width > width_out:
                    start = (width - width_out) // 2
                    end = start + width_out
                    x_s = x_s[..., start:end]
            if flip:
                x_s = np.flip(x_s, -1)
        else:
            if missing_value_policy == "sparse":
                return None
            x_s = np.nan * np.ones(((self.n_channels,) + crop_size), dtype=np.float32)

        # If we are here we're not returning None.
        if missing_value_policy == "sparse":
            missing_value_policy = "missing"

        x_s = torch.tensor(x_s.copy(), dtype=torch.float32)
        if missing_value_policy == "masked":
            x_s = MaskedTensor(x_s)
        x_s = self.replace_missing(x_s, missing_value_policy, rng)

        return x_s
