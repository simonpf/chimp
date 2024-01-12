"""
chimp.data.reference
===================

This module provides functions for loading CHIMP reference data.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,  List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
import torch
import xarray as xr

from chimp.data.source import DataSource
from chimp.data.utils import scale_slices


def find_random_scene(
    reference_dataset,
    path: Path,
    rng: np.random.Generator,
    multiple: int = 4,
    scene_size: int = 256,
    quality_threshold: float = 0.8,
    valid_fraction: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Finds a random scene in the reference data that has given minimum
    fraction of values of values exceeding a given RQI threshold.

    Args:
        reference_dataset: Reference dataset object reprsenting the reference
            data to load.
        path: The path of the reference data file from which to sample a random
            scene.
        rng: A numpy random generator instance to randomize the scene search.
        multiple: Limits the scene position to coordinates that a multiples
            of the given value.
        quality_threshold: If the reference dataset has a quality index,
            all reference data pixels below the given threshold will considered
            invalid outputs.
        valid_fraction: The minimum amount of valid samples in the extracted
            region.

    Return:
        A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
        of the random crop.
    """
    with xr.open_dataset(path) as data:

        if reference_dataset.quality_index is not None:
            qi = data[reference_dataset.quality_index].data
        else:
            qi = np.isfinite(data[reference_dataset.targets[0].name].data)

        found = False
        count = 0
        while not found:
            if count > 20:
                return None
            count += 1
            n_rows, n_cols = qi.shape
            i_start = rng.integers(0, (n_rows - scene_size) // multiple)
            i_end = i_start + scene_size // multiple
            j_start = rng.integers(0, (n_cols - scene_size) // multiple)
            j_end = j_start + scene_size // multiple

            i_start = i_start * multiple
            i_end = i_end * multiple
            j_start = j_start * multiple
            j_end = j_end * multiple

            row_slice = slice(i_start, i_end)
            col_slice = slice(j_start, j_end)

            if (qi[row_slice, col_slice] > quality_threshold).mean() > valid_fraction:
                found = True

    return (i_start, i_end, j_start, j_end)


@dataclass
class RetrievalTarget:
    """
    This dataclass holds properties of retrieval targets provided
    by a reference dataset.
    """

    name: str
    lower_limit: Optional[float] = None


ALL_REFERENCE_DATA = {}


@dataclass
class ReferenceData(DataSource):
    """
    This dataclass holds properties of reference datasets.
    """

    name: str
    scale: int
    targets: List[RetrievalTarget]
    quality_index: str

    def __init__(
        self, name: str, scale: int, targets: list[RetrievalTarget], quality_index: str
    ):
        super().__init__(name)
        ALL_REFERENCE_DATA[name] = self
        self.name = name
        self.scale = scale
        self.targets = targets
        self.quality_index = quality_index

    def find_files(self, path: Path) -> List[Path]:
        """
        Find reference data files.

        Args:
            path: Path to the folder containing the training data.

        Return:
            A list of found reference data files.
        """
        pattern = "*????????_??_??.nc"
        reference_files = sorted(
            list((path / self.name).glob(pattern))
        )
        return reference_files


    def load_sample(
            self,
            path: Path,
            crop_size: int,
            base_scale,
            slices: Tuple[int, int, int, int],
            rng: np.random.Generator,
            rotate: Optional[float] = None,
            flip: Optional[bool] = None,
            quality_threshold: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """
        Load sample from reference data.

        Args:
            path: The paeh of the reference data file from which to load the
                sample.
            crop_size: The size of the input slice to load to load.
            base_scale: The scale with respect to which the slices are defined.
            slices: Tuple of ints defining the subset of reference data to load.
            rng: A numpy random generator object to use to generate random numbers.
            rotate: An optional float indicating the degrees by which to rotate
                the input.
            flip: Whether or not the input should be flipped.

        Return:
            A dictionary mapping retrieval target names to corresponding tensors.
        """
        from pytorch_retrieve.tensors.masked_tensor import MaskedTensor
        rel_scale = self.scale / base_scale
        if isinstance(crop_size, int):
            crop_size = (crop_size,) * self.n_dims
        crop_size = tuple((int(size / rel_scale) for size in crop_size))
        row_slice, col_slice = scale_slices(slices, rel_scale)

        y = {}
        with xr.open_dataset(path) as data:

            if self.quality_index is not None:
                qual = data[self.quality_index]
                qual = qual.data[row_slice, col_slice]
                invalid = qual < quality_threshold
            else:
                invalid = None

            for target in self.targets:
                y_t = data[target.name].data[row_slice, col_slice]
                if not np.issubdtype(y_t.dtype, np.floating):
                    y_t = y_t.astype(np.int64)

                # Set window size here if it is None
                if crop_size is None:
                    crop_size = y_t.shape[-2:]

                if target.lower_limit is not None:
                    y_t[y_t < 0] = np.nan
                    small = y_t < target.lower_limit
                    rnd = rng.uniform(-5, -3, small.sum())
                    y_t[small] = 10**rnd

                if invalid is not None:
                    y_t[invalid] = np.nan

                if rotate is not None:
                    y_t = ndimage.rotate(
                        y_t, rotate, order=0, axes=(-2, -1), reshape=False, cval=np.nan
                    )

                    height = y_t.shape[-2]
                    if height > crop_size[0]:
                        d_l = (height - crop_size[0]) // 2
                        d_r = d_l + crop_size[0]
                        y_t = y_t[..., d_l:d_r, :]
                    width = y_t.shape[-1]
                    if width > crop_size[1]:
                        d_l = (width - crop_size[1]) // 2
                        d_r = d_l + crop_size[1]
                        y_t = y_t[..., d_l:d_r]

                if flip:
                    y_t = np.flip(y_t, -1)

                mask = torch.tensor(np.isnan(y_t))
                tensor = torch.tensor(y_t.copy())
                y[target.name] = MaskedTensor(tensor, mask=mask)
        return y


def get_reference_data(name: Union[str, ReferenceData]) -> ReferenceData:
    """
    Retrieve reference dataset by name.

    Args:
        name: The name of a dataset.

    Return:
        A ReferenceData object that can be used to load reference data
        from the requested dataset.
    """
    from . import baltrad
    from . import mrms

    if isinstance(name, DataSource):
        return name
    if name in ALL_REFERENCE_DATA:
        return ALL_REFERENCE_DATA[name]

    raise ValueError(
        f"The reference data '{name}' is currently not available. Available "
        f" reference datasets are {list(ALL_REFERENCE_DATA.keys())}."
    )
