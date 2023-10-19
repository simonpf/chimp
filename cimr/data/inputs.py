"""
cimr.data.inputs
================

Defines a dataclass to represent input data sources.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
from quantnn.normalizer import Normalizer, MinMaxNormalizer
import xarray as xr


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


@dataclass
class Input:
    """
    Record holding the paths of the files for a single training
    sample.
    """
    name: str
    scale: int
    variables: Union[str, List[str]]
    normalizer: Normalizer
    mean : Optional[np.array] = None

    @property
    def n_channels(self):
        return len(self.normalizer.stats)

    def load_sample(
            self,
            input_file: Path,
            crop_size: int,
            rng: np.random.Generator
            missing_input_policy: str
    ):
        # Input is missing: Apply missing value policy.
        if input_file is None:
            x_s = generate_input(
                inpt,
                tuple([size // self.scales[inpt.name] for size in window_size]),
                missing_input_policy,
                rng
            )
            x[inpt.name] = x_s
            continue

        scl = self.scales[inpt.name]
        if slices is not None:
            row_slice = slice(int(i_start / scl), int(i_end / scl))
            col_slice = slice(int(j_start / scl), int(j_end / scl))
        else:
            row_slice = slice(0, None)
            col_slice = slice(0, None)
        with xr.open_dataset(files[input_ind + 1]) as data:
            vars = inpt.variables
            if isinstance(vars, str):
                x_s = data[vars].data[..., row_slice, col_slice]
                # Expand dims in case of single-channel inputs.
                if x_s.ndim < 3:
                    x_s = x_s[None]
            else:
                x_s = np.stack(
                    [data[vrbl].data[row_slice, col_slice] for vrbl in vars]
                )
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
                if height > window_size[0] // scl:
                    d_l = (height - window_size[0] // scl) // 2
                    d_r = d_l + window_size[0] // scl
                    x_s = x_s[..., d_l:d_r, :]
                width = x_s.shape[-1]
                if width > window_size[1] // scl:
                    d_l = (width - window_size[1] // scl) // 2
                    d_r = d_l + window_size[1] // scl
                    x_s = x_s[..., d_l:d_r]
            if flip:
                x_s = np.flip(x_s, -1)

            if self.normalize:
                x_s = inpt.normalizer(x_s)

###############################################################################
# GMI
###############################################################################

NORMALIZER_GMI = MinMaxNormalizer(np.ones((12, 1, 1)), feature_axis=0)
NORMALIZER_GMI.stats = {
    0: (150, 330),
    1: (70, 330),
    2: (160, 330),
    3: (80, 330),
    4: (170, 320),
    5: (170, 320),
    6: (110, 320),
    7: (80, 310),
    8: (80, 320),
    9: (80, 310),
    10: (70, 310),
    11: (80, 300),
    12: (80, 300),
}
GMI = Input("gmi", 8, "tbs", NORMALIZER_GMI)

###############################################################################
# MHS
###############################################################################

NORMALIZER_MHS = MinMaxNormalizer(np.ones((5, 1, 1)), feature_axis=0)
NORMALIZER_MHS.stats = {
    0: (80, 310),
    1: (80, 310),
    2: (100, 290),
    3: (90, 290),
    4: (90, 300),
}
MHS = Input("mhs", 16, "tbs", NORMALIZER_MHS)


###############################################################################
# ATMS
###############################################################################

NORMALIZER_ATMS = MinMaxNormalizer(np.ones((9, 1, 1)), feature_axis=0)
NORMALIZER_ATMS.stats = {
    0: (130, 320),
    1: (140, 320),
    2: (140, 320),
    3: (90, 320),
    4: (90, 310),
    5: (90, 310),
    6: (100, 300),
    7: (100, 290),
    8: (110, 290),
}
MEANS_ATMS = np.array([
    238.66225432, 231.23515187, 254.60181577, 268.60674832,
    266.81132495, 263.05214856, 258.46545914, 251.43006749,
    244.87598689
])
ATMS = Input("atms", 16, "tbs", NORMALIZER_ATMS)

###############################################################################
# SSMIS
###############################################################################

NORMALIZER_SSMIS = MinMaxNormalizer(np.ones((11, 1, 1)), feature_axis=0)
NORMALIZER_SSMIS.stats = {
    0: (100, 330),
    1: (70, 330),
    2: (110, 320),
    3: (170, 320),
    4: (120, 310),
    5: (80, 290),
    6: (80, 290),
    7: (70, 320),
    8: (70, 310),
    9: (70, 320),
    10: (60, 310),
}
SSMIS = Input("ssmis", 8, "tbs", NORMALIZER_SSMIS)

###############################################################################
# AMSR2
###############################################################################

NORMALIZER_AMSR2 = MinMaxNormalizer(np.ones((11, 1, 1)), feature_axis=0)
NORMALIZER_AMSR2.stats = {
    0: (60, 330),
    1: (70, 330),
    2: (80, 330),
    3: (60, 330),
    4: (150, 320),
    5: (100, 310),
    6: (90, 320),
    7: (90, 320),
    8: (50, 330),
    9: (50, 320),
    10: (50, 320),
    11: (50, 320),
}
AMSR2 = Input("amsr2", 8, "tbs", NORMALIZER_AMSR2)

###############################################################################
# CPCIR
###############################################################################

NORMALIZER_CPCIR = MinMaxNormalizer(np.ones((1, 1, 1)), feature_axis=0)
NORMALIZER_CPCIR.stats = {
    0: (170, 320)
}
CPCIR = Input("cpcir", 4, "tbs", NORMALIZER_CPCIR)
