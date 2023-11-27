"""
chimp.data.training_data
=======================

Interface classes for loading the CHIMP training data.
"""
from dataclasses import dataclass
from datetime import datetime
from math import ceil
import os
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import fft
from scipy import ndimage
from quantnn.normalizer import MinMaxNormalizer
from quantnn.packed_tensor import PackedTensor
import torch
from torch import nn
from torch.utils.data import IterableDataset
import torch.distributed as dist
import xarray as xr
import pandas as pd


from chimp import data
from chimp.definitions import MASK
from chimp.utils import get_date
from chimp.data import get_reference_data
from chimp.data import get_input, get_reference_data
from chimp.data import input, reference

###############################################################################
# Normalizer objects
###############################################################################

NORMALIZER_GEO = MinMaxNormalizer(np.ones((12, 1, 1)), feature_axis=0)
NORMALIZER_GEO.stats = {
    0: (0.0, 110.0),
    1: (0.0, 130),
    2: (0.0, 110),
    3: (200, 330),
    4: (210, 260),
    5: (200, 270),
    6: (200, 300),
    7: (220, 270),
    8: (200, 300),
    9: (200, 300),
    10: (200, 280),
}


NORMALIZER_VISIR = MinMaxNormalizer(np.ones((5, 1, 1)), feature_axis=0)
NORMALIZER_VISIR.stats = {
    0: (0.0, 320.0),
    1: (0.0, 320.0),
    2: (0.0, 320.0),
    3: (180, 310),
    4: (180, 310),
}


NORMALIZER_MW_90 = MinMaxNormalizer(np.ones((2, 1, 1)), feature_axis=0)
NORMALIZER_MW_90.stats = {
    0: (150, 300),
    1: (150, 300),
}

NORMALIZER_MW_160 = MinMaxNormalizer(np.ones((2, 1, 1)), feature_axis=0)
NORMALIZER_MW_160.stats = {
    0: (150, 300),
    1: (150, 300),
}

NORMALIZER_MW_183 = MinMaxNormalizer(np.ones((5, 1, 1)), feature_axis=0)
NORMALIZER_MW_183.stats = {
    0: (190, 290),
    1: (190, 290),
    2: (190, 290),
    3: (190, 290),
    4: (190, 290),
}



###############################################################################
# Loader functions for the difference input types.
###############################################################################

def collate_recursive(sample, batch=None):
    """
    Recursive collate function that descends into tuple, lists and
    dicts, and collects tensors and collects tensors and None values
    into lists.

    Args:
        sample: Sample may be a tuple, list or dict or any nested
            combination of those containing either tensors or None.
        batch: Previously collected samples.

    Return:
        The returned value has the same structure as ``sample`` but the
        leaf values are replaced by lists containing the original leaf
        values. If batch is not ``None`` then these lists also contain
        the previously collected leaf-lists from batch.
    """
    if batch is None:
        if isinstance(sample, tuple):
            return tuple([collate_recursive(sample_t, None) for sample_t in sample])
        elif isinstance(sample, list):
            return [collate_recursive(sample_t, None) for sample_t in sample]
        elif isinstance(sample, dict):
            return {
                k: collate_recursive(sample_k) for k, sample_k in sample.items()
            }
        elif isinstance(sample, torch.Tensor):
            return [sample]
        elif sample is None:
            return [sample]
        else:
            try:
                return [torch.as_tensor(sample)]
            except ValueError:
                pass
    else:
        if isinstance(sample, tuple):
            return tuple([
                collate_recursive(sample_t, batch_t)
                 for sample_t, batch_t in zip(sample, batch)
            ])
        elif isinstance(sample, list):
            return [
                collate_recursive(sample_t, batch_t)
                 for sample_t, batch_t in zip(sample, batch)
            ]
        elif isinstance(sample, dict):
            return {
                k: collate_recursive(sample_k, batch[k])
                for (k, sample_k) in sample.items()
            }
        elif isinstance(sample, torch.Tensor):
            return batch + [sample]
        elif sample is None:
            return batch + [sample]
        else:
            try:
                return batch + [torch.as_tensor(sample)]
            except ValueError:
                pass
    raise ValueError(
        "Encountered invalid type '%s' in collate function.",
        type(sample)
    )


def is_tensor(x):
    """
    Helper function to determine whether an object is a tensor.
    """
    return isinstance(x, torch.Tensor)


def stack(batch):
    """
    Stack collected samples into sparse tensors.

    batch:
        An optionally nested structure of tuples, list, and dicts
        with lists of tensors and None values as leaves.

    Return:
        A copy of batch but with all leaves replaced by corresponding
        packed tensors.
    """
    if isinstance(batch, tuple):
        return tuple([stack(batch_t) for batch_t in batch])
    elif isinstance(batch, list):
        try:
            if all(map(is_tensor, batch)):
                return torch.stack(batch)
            batch = PackedTensor.stack(batch)
            return batch
        except ValueError:
            return [stack(batch_t) for batch_t in batch]
    elif isinstance(batch, dict):
        return {
            k: stack(batch_k) for k, batch_k in batch.items()
        }
    raise ValueError(
        "Encountered invalid type '%s' in stack function.",
        type(batch)
    )


def sparse_collate(samples):
    """
    Collate a list of samples into a batch of packed tensors.

    Args:
        samples: A list of samples to collate into a batch.

    Return:
        A batch of input samples with all input samples collected
        into PackedTensors.
    """
    batch = None
    for sample in samples:
        batch = collate_recursive(sample, batch)
    return stack(batch)




def generate_input(
        inpt: data.Input,
        size: Tuple[int],
        policy: str,
        rng: np.random.Generator,
):
    """
    Generate input values for missing inputs.

    Args:
        inpt: The input object for which to generate the input.
        size: Side-length of the quadratic input array.
        value: The value to fill the array with.
        policy: The policy to use for the data generation.
        rng: The random generator object to use to create random
            arrays.

    Return:
        An numpy.ndarray containing replacement data.
    """
    if policy == "sparse":
        return None
    elif policy == "random":
        return rng.normal(size=(inpt.n_channels,) + size)
    elif policy == "mean":
        return inpt.normalizer(inpt.mean * np.ones(
            shape=(inpt.n_channels, size, size),
            dtype="float32"
        ))
    elif policy == "missing":
        return inpt.normalizer(np.nan * np.ones(
            shpare=(inpt.n_channels, size, size),
            dtype="float32"
        ))

    raise ValueError(
        f"Missing input policy '{policy}' is not known. Choose between 'random'"
        " 'mean' and 'constant'. "
    )


@dataclass
class SampleRecord:
    """
    Record holding the paths of the files for a single training
    sample.
    """
    radar: Path = None
    geo: Path = None
    mw: Path = None
    visir: Path = None

    def has_input(self, sources):
        """
        Determine if sample has input from any of the given sources.

        Args:
            sources: A list of sources to require.

        Return:
            Bool if the sample has corresponding input data from any of
            the given sources.
        """
        has_input = False
        for source in sources:
            if getattr(self, source) is not None:
                has_input = True
        return has_input




class SingleStepDataset:
    """
    Dataset class for the CHIMP training data.

    Implements the PyTorch Dataset interface.
    """
    def __init__(
        self,
        folder,
        inputs,
        reference_data,
        sample_rate=1,
        sequence_length=1,
        normalize=True,
        window_size=128,
        start_time=None,
        end_time=None,
        quality_threshold=0.8,
        augment=True,
        missing_value_policy="sparse",
        time_step=None,
        validation=False
    ):
        """
        Args:
            folder: The root folder containing the training data.
            reference_data: Name of the folder containing the reference
                data.
            inputs: List of names of inputs specifying the input data.
            sample_rate: How often each scene should be sampled per epoch.
            normalize: Whether inputs should be normalized.
            window_size: Size of the training scenes.
            start_time: Start time of time interval to which to restrict
                training data.
            end_time: End time of a time interval to which to restrict the
                training data.
            quality_threshold: Threshold for radar quality index used to mask
                the radar measurements.
            augment: Whether to apply random transformations to the training
                inputs.
            missing_value_policy: A string indicating how to handle missing input
                data. Options:
                    'random': Missing data is replaced with Gaussian noise.
                    'mean' Missing value is replaced with the mean of the
                    input data.
                    'missing': Missing data is replaced with NANs
                    'sparse': Instead of a tensor 'None' is returned.
            time_step: Minimum time step between consecutive reference samples.
                Can be used to sub-sample the reference data.
            validation: If 'True' sampling will reproduce identical scenes.
        """
        self.folder = Path(folder)

        self.reference_data = get_reference_data(reference_data)

        if inputs is None:
            inputs = ["visir", "geo", "mw"]
        self.inputs = [get_input(inpt) for inpt in inputs]

        self.n_inputs = len(inputs)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.quality_threshold = quality_threshold
        self.augment = augment
        self.missing_value_policy = missing_value_policy

        pattern = "*????????_??_??.nc"

        # Find all reference files and select those within time range.
        reference_files = sorted(
            list((self.folder / self.reference_data.name).glob(pattern))
        )
        times = np.array(list(map(get_date, reference_files)))
        if start_time is not None and end_time is not None:
            indices = (times >= start_time) * (times < end_time)
            times = times[indices]
            reference_files = [reference_files[i] for i in np.where(indices)[0]]

        if time_step is not None:
            d_t = np.timedelta64(time_step.seconds, "s")
            d_t_files = times[1] - times[0]
            subsample = int(d_t / d_t_files)
            reference_files = reference_files[::subsample]

        samples = {
            time: [ref_file,] + [None,] * self.n_inputs for
            time, ref_file in zip(times, reference_files)
        }

        self.base_scale = self.reference_data.scale


        self.scales = {}
        self.max_scale = 0
        for src_ind, inpt in enumerate(self.inputs):
            src_files = sorted(list((self.folder / inpt.name).glob(pattern)))
            times = np.array(list(map(get_date, src_files)))
            scale = None
            for time, src_file in zip(times, src_files):

                if scale is None:
                    with xr.open_dataset(src_file) as src_data:
                        scale = inpt.scale // self.base_scale
                        self.scales[inpt.name] = scale
                        self.max_scale = max(self.max_scale, scale)

                sample = samples.get(time)
                if sample is not None:
                    sample[src_ind + 1] = src_file

        self.keys = np.array(list(samples.keys()))
        self.samples = np.array(list(samples.values()))
        self.window_size = window_size

        self.validation = validation
        if self.validation:
            self.augment = False
        self.init_rng()

    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        if self.validation:
            seed = 1234
        else:
            seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def load_sample(
            self,
            files: Tuple[Path],
            slices: Optional[Tuple[int]],
            window_size: int,
            forecast: bool = False,
            rotate: Optional[float] = None,
            flip: bool = False,
    ):
        """
        Load training sample.

        Args:
            files: A list containing the reference-data and input files
                 containing the data from which to load this sample.
            slices: Tuple ``(i_start, i_end, j_start, j_end)`` defining
                defining the crop of the domain. If set to 'None', the full
                domain is loaded.
            window_size: The window size specified with respect to the
                reference data.
            forecast: If 'True', no input data will be loaded and all inputs
                will be set to None.
            rotate: If provided, should be float specifying the degree by which
                to rotate the input.
            flip: If 'True', input will be flipped along the last axis.

        Return:
            A tuple ``(x, y)`` containing two dictionaries ``x`` and ``y``
            with ``x`` containing the training inputs and ``y`` the
            corresponding outputs.
        """
        if isinstance(window_size, int):
            window_size = (window_size,) * 2

        if slices is not None:
            i_start, i_end, j_start, j_end = slices
            row_slice = slice(i_start, i_end)
            col_slice = slice(j_start, j_end)
        else:
            row_slice = slice(0, None)
            col_slice = slice(0, None)

        # Load reference data.
        y = {}
        with xr.open_dataset(files[0]) as data:
            qual = data[self.reference_data.quality_index]
            qual = qual.data[row_slice, col_slice]
            invalid = qual < self.quality_threshold
            for target in self.reference_data.targets:

                y_t = data[target.name].data[row_slice, col_slice]
                if not np.issubdtype(y_t.dtype, np.floating):
                    y_t = y_t.astype(np.int64)

                # Set window size here if it is None
                if window_size is None:
                    window_size = y_t.shape[-2:]

                if target.lower_limit is not None:
                    y_t[y_t < 0] = np.nan
                    small = y_t < target.lower_limit
                    rnd = self.rng.uniform(-5, -3, small.sum())
                    y_t[small] = 10 ** rnd
                y_t[invalid] = MASK
                if rotate is not None:
                    y_t = ndimage.rotate(
                        y_t,
                        rotate,
                        order=0,
                        axes=(-2, -1),
                        reshape=False,
                        cval=np.nan
                    )

                    height = y_t.shape[-2]
                    if height > window_size[0]:
                        d_l = (height - window_size[0]) // 2
                        d_r = d_l + window_size[0]
                        y_t = y_t[..., d_l:d_r, :]
                    width = y_t.shape[-1]
                    if width > window_size[1]:
                        d_l = (width - window_size[1]) // 2
                        d_r = d_l + window_size[1]
                        y_t = y_t[..., d_l:d_r]

                if flip:
                    y_t = np.flip(y_t, -1)

                if np.issubdtype(y_t.dtype, np.floating):
                    y_t = np.nan_to_num(y_t, nan=MASK, copy=True)
                y[target.name] = torch.tensor(y_t.copy())

        # Load input data.
        x = {}
        for input_ind, inpt in enumerate(self.inputs):
            input_file = files[input_ind + 1]
            x_s = inpt.load_sample(
                input_file,
                window_size,
                self.base_scale,
                slices,
                self.rng,
                self.missing_value_policy,
                rotate=rotate,
                flip=flip
            )
            x[inpt.name] = x_s

        return x, y


    def __len__(self):
        """Number of samples in dataset."""
        return len(self.samples) * self.sample_rate


    def __getitem__(self, index):
        """Return ith training sample."""
        n_starts = len(self.samples)
        scene_index = index % n_starts

        # We load a larger window when input is rotated to avoid
        # missing values.
        if self.augment:
            window_size = int(1.42 * self.window_size)
            rem = window_size % self.max_scale
            if rem != 0:
                window_size += self.max_scale - rem
            ang = -180 + 360 * self.rng.random()
            flip = self.rng.random() > 0.5
        else:
            window_size = self.window_size
            ang = None
            flip = False

        slices = reference.find_random_scene(
            self.reference_data,
            self.samples[scene_index][0],
            self.rng,
            multiple=4,
            window_size=window_size,
            qi_thresh=self.quality_threshold
        )


        x, y = self.load_sample(
            self.samples[scene_index],
            slices,
            self.window_size,
            rotate=ang,
            flip=flip
        )

        has_input = any((x[inpt.name] is not None for inpt in self.inputs))
        has_output = any (
            (y[target.name] is not None for target in self.reference_data.targets)
        )

        if not has_input or not has_output:
            new_index = self.rng.integers(0, len(self))
            return self[new_index]

        return x, y


    def plot(self, key, domain):

        sample = self.samples[key]

        keys = list(domain.keys())
        domain = domain[min(keys)]

        crs = domain.to_cartopy_crs()
        extent = domain.area_extent
        extent = (extent[0], extent[2], extent[1], extent[3])
        cmap = "plasma"

        f = plt.figure(figsize=(20, 20))
        gs = GridSpec(4, 5)
        axs = np.array([[
            f.add_subplot(gs[i, j], projection=crs) for j in range(5)
            ] for i in range(4)
        ])

        radar_data = xr.load_dataset(sample.radar)

        ax = axs[0, 0]
        dbz = radar_data.dbz.data.copy()
        dbz[radar_data.qi.data < self.quality_threshold] = np.nan
        img = ax.imshow(dbz, extent=extent, vmin=-20, vmax=40, cmap=cmap)
        ax.coastlines(color="w")

        if sample.geo is not None:
            channels = [1, 4, 7, 11]
            data_geo = xr.load_dataset(sample.geo)
            for i, ch in enumerate(channels):
                ax = axs[0, 1 + i]
                ax.imshow(data_geo[f"geo_{ch:02}"].data, extent=extent, cmap=cmap)
                ax.coastlines(color="w")

        if sample.visir is not None:
            data_visir = xr.load_dataset(sample.visir)
            for i in range(5):
                ax = axs[1, i]
                ax.imshow(data_visir[f"visir_{(i + 1):02}"].data, extent=extent, cmap=cmap)
                ax.coastlines(color="w")

        if sample.mw is not None:
            data_mw = xr.load_dataset(sample.mw)
            if "mw_90" in data_mw:
                for i in range(2):
                    ax = axs[2, i]
                    ax.imshow(
                        data_mw["mw_90"].data[..., i],
                        extent=extent,
                        cmap=cmap
                    )
                    ax.coastlines(color="w")
            if "mw_160" in data_mw:
                for i in range(2):
                    ax = axs[2, i + 2]
                    ax.imshow(
                        data_mw["mw_160"].data[..., i],
                        extent=extent,
                        cmap=cmap
                    )
                    ax.coastlines(color="w")

            if "mw_183" in data_mw:
                for i in range(5):
                    ax = axs[3, i]
                    ax.imshow(
                        data_mw["mw_183"].data[..., i],
                        extent=extent,
                        cmap=cmap
                    )
                    ax.coastlines(color="w")

        return f, axs

    def make_animator(self, start_time, end_time):
        indices = (self.keys >= start_time) * (self.keys <= end_time)
        keys = self.keys[indices]

        crs = domain.to_cartopy_crs()
        extent = domain.area_extent
        extent = (extent[0], extent[2], extent[1], extent[3])

        f = plt.figure(figsize=(10, 12))
        gs = GridSpec(2, 2)
        axs = np.array(
            [
                [f.add_subplot(gs[i, j], projection=crs) for j in range(2)]
                for i in range(2)
            ]
        )

        ax = axs[0, 0]
        ax.set_title("Radar")

        ax = axs[0, 1]
        ax.set_title("geo 12u")

        ax = axs[1, 0]
        ax.set_title("VISIR")

        ax = axs[1, 1]
        ax.set_title("MHS (89 GHz)")

        def animator(frame):

            sample = self.samples[keys[frame]]

            radar_data = xr.load_dataset(sample.radar)
            ax = axs[0, 0]
            dbz = radar_data.dbz.data.copy()
            dbz[radar_data.qi.data < self.quality_threshold] = np.nan
            img = ax.imshow(dbz, extent=extent, vmin=-20, vmax=20)
            ax.coastlines(color="grey")

            ax = axs[0, 1]
            if sample.geo is not None:
                geo_data = xr.load_dataset(sample.geo)
                ax.imshow(geo_data.channel_10, extent=extent)
                ax.coastlines(color="grey")
            else:
                img = np.nan * np.ones((2, 2))
                img = ax.imshow(img, extent=extent)
                ax.coastlines(color="grey")

            ax = axs[1, 0]
            if sample.avhrr is not None:
                avhrr_data = xr.load_dataset(sample.avhrr)
                img = ax.imshow(avhrr_data.channel_5, extent=extent)
                ax.coastlines(color="grey")
            else:
                img = np.nan * np.ones((2, 2))
                img = ax.imshow(img, extent=extent)
                ax.coastlines(color="grey")

            ax = axs[1, 1]
            if sample.mhs is not None:
                mhs_data = xr.load_dataset(sample.mhs)
                img = ax.imshow(mhs_data.channel_01, extent=extent)
                ax.coastlines(color="grey")
            else:
                img = np.nan * np.ones((2, 2))
                img = ax.imshow(img, extent=extent)
                ax.coastlines(color="grey")

            return [img]

        return f, axs, animator

    def load_full_data(self, key):
        with xr.open_dataset(self.samples[key].radar) as data:
            shape = data.dbz.data.shape
            y = data.dbz.data.copy()
            y[data.qi.data < self.quality_threshold] = np.nan

        y = torch.as_tensor(y, dtype=torch.float)
        shape = y.shape

        x = {}

        # VISIR data
        if self.samples[key].visir is not None:
            with xr.open_dataset(self.samples[key].visir) as data:
                load_visir_obs(x, data, normalize=self.normalize)
        else:
            x["visir"] = -1.5 * torch.ones(
                (5,) + shape,
                dtype=torch.float
            )

        shape = tuple([n // 2 for n in y.shape])
        if self.samples[key].geo is not None:
            with xr.open_dataset(self.samples[key].geo) as data:
                load_geo_obs(x, data, normalize=self.normalize)
        else:
            x["geo"] = torch.as_tensor(
                -1.5 * np.ones((11,) + shape),
                dtype=torch.float
            )

        shape = tuple([n // 4 for n in y.shape])
        # Microwave data
        if self.samples[key].mw is not None:
            with xr.open_dataset(self.samples[key].mw) as data:
                load_microwave_obs(x, data, normalize=self.normalize
                )
        else:
            x["mw_90"] = torch.as_tensor(
                -1.5 * np.ones((2,) + shape),
                dtype=torch.float,
            )
            x["mw_160"] = torch.as_tensor(
                -1.5 * np.ones((2,) + shape),
                dtype=torch.float,
            )
            x["mw_183"] = torch.as_tensor(
                -1.5 * np.ones((5,) + shape),
                dtype=torch.float,
            )

        return x, y

    def pad_input(self, x, multiple=32):
        """
        Pad retrieval input to a size that is a multiple of a given n.

        Args:
            x: A dict containing the input to pad.
            multiple: The number n.

        Return:
            A tuple ``(x_pad, y_slice, x_slice)`` containing the padded input
            ``x_pad`` and column- and row-slices to extract the unpadded
            output.
        """
        input_visir = x["visir"]
        input_geo = x["geo"]
        input_mw_90 = x["mw_90"]
        input_mw_160 = x["mw_160"]
        input_mw_183 = x["mw_183"]

        shape = input_visir.shape[-2:]

        padding_y = np.ceil(shape[0] / multiple) * multiple - shape[0]
        padding_y_l = padding_y // 2
        padding_y_r = padding_y - padding_y_l
        padding_x = np.ceil(shape[1] / multiple) * multiple - shape[1]
        padding_x_l = padding_x // 2
        padding_x_r = padding_x - padding_x_l
        padding = (
            int(padding_x_l),
            int(padding_x_r),
            int(padding_y_l),
            int(padding_y_r),
        )

        slice_x = slice(int(padding_x_l), int(-padding_x_r))
        slice_y = slice(int(padding_y_l), int(-padding_y_r))

        input_visir = nn.functional.pad(input_visir, padding, mode="constant")
        x["visir"] = input_visir

        padding_y = padding_y // 2
        padding_y_l = padding_y // 2
        padding_y_r = padding_y - padding_y_l
        padding_x = padding_x // 2
        padding_x_l = padding_x // 2
        padding_x_r = padding_x - padding_x_l
        padding = (
            int(padding_x_l),
            int(padding_x_r),
            int(padding_y_l),
            int(padding_y_r),
        )
        input_geo = nn.functional.pad(input_geo, padding, mode="constant")
        x["geo"] = input_geo

        padding_y = padding_y // 2
        padding_y_l = padding_y // 2
        padding_y_r = padding_y - padding_y_l
        padding_x = padding_x // 2
        padding_x_l = padding_x // 2
        padding_x_r = padding_x - padding_x_l
        padding = (
            int(padding_x_l),
            int(padding_x_r),
            int(padding_y_l),
            int(padding_y_r),
        )

        input_mw_90 = nn.functional.pad(input_mw_90, padding, mode="constant")
        x["mw_90"] = input_mw_90
        input_mw_160 = nn.functional.pad(input_mw_160, padding, mode="constant")
        x["mw_160"] = input_mw_160
        input_mw_183 = nn.functional.pad(input_mw_183, padding, mode="constant")
        x["mw_183"] = input_mw_183

        return x, slice_y, slice_x


    def full_domain(self, start_time=None, end_time=None):

        valid = np.ones(self.keys.size, dtype="bool")
        if start_time is not None:
            valid = valid * (self.keys >= start_time)
        if end_time is not None:
            valid = valid * (self.keys <= end_time)

        indices = np.where(valid)[0]
        time_steps = np.argsort(self.keys[indices])

        for time_step in time_steps:
            x, y = self.load_sample(
                self.samples[indices[time_step]],
                None,
                None
            )
            time = self.keys[indices[time_step]]
            x = sparse_collate([x])
            yield time, x, y


    def get_forecast_input(self, forecast_time, n_obs):
        """
        Get input for a forecast.

        Args:
            forecast_time: The time a which the forecast should
                be initiated.
            n_obs: The number of observations previous to the
                forecast.
        """
        input_indices = self.keys <= forecast_time
        input_keys = self.keys[input_indices][-n_obs:]

        if len(input_keys) < n_obs:
            return None
        if np.any(np.diff(input_keys) > np.timedelta64(20, "m")):
            return None
        if input_keys[-1] != forecast_time:
            return None

        inputs = []

        for key in input_keys:
            x, y = self.load_full_data(key)
            x, slice_y, slice_x = self.pad_input(x, multiple=64)

            has_input = False

            missing_fraction = (x["geo"] < -1.4).float().mean()
            if missing_fraction > 0.95:
                x["geo"] = None
            else:
                has_input = True

            missing_fraction = (x["visir"] < -1.4).float().mean()
            if missing_fraction > 0.95:
                x["visir"] = None
            else:
                has_input = True

            missing_fraction = 0
            keys_mw = ["mw_90", "mw_160", "mw_183"]
            for mw in keys_mw:
                missing_fraction += (x[mw] < -1.4).float().mean()
            if missing_fraction / 3 > 0.95:
                for mw in keys_mw:
                    x[mw] = None
            else:
                has_input = True

            if not has_input:
                return None

            x = sparse_collate([x])
            inputs.append((x, y, slice_y, slice_x, key))
        return inputs


class CHIMPPretrainDataset(SingleStepDataset):
    """
    Dataset class for the CHIMP training data.

    Implements the PyTorch Dataset interface.
    """
    def __init__(
        self,
        folder,
        reference_data="radar",
        inputs=None,
        sample_rate=1,
        sequence_length=1,
        normalize=True,
        window_size=128,
        start_time=None,
        end_time=None,
        quality_threshold=0.8,
        augment=True,
        missing_value_policy="sparse",
        time_step=None
    ):
        super().__init__(
            folder,
            reference_data=reference_data,
            inputs=inputs,
            sample_rate=sample_rate,
            sequence_length=sequence_length,
            normalize=normalize,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            quality_threshold=quality_threshold,
            augment=augment,
            missing_value_policy=missing_value_policy,
            time_step=time_step
        )
        samples_by_input = [[] for _ in inputs]
        for scene_index in range(len(self.samples)):
            sample = self.samples[scene_index]
            for i in range(len(inputs)):

                # Input not available at time step
                if sample[1 + i] is None:
                    continue

                with xr.open_dataset(sample[1 + i]) as data:
                    if "swath_centers" not in data.dims:
                        samples_by_input[i].append(scene_index)
                    else:
                        if data.swath_centers.size > 0:
                            samples_by_input[i].append(scene_index)

        most_obs = max(map(len, samples_by_input))
        total_samples = len(inputs) * most_obs

        new_starts = []
        for i in range(len(inputs)):
            new_starts.append(
                self.rng.choice(
                    samples_by_input[i],
                    most_obs,
                    replace=True
                )
            )
        self.sequence_starts = np.concatenate(new_starts)
        self.obs_per_input = most_obs

    def __getitem__(self, index):
        """Return ith training sample."""

        scene_index = self.sequence_starts[index // self.sample_rate]
        input_index = index // self.obs_per_input
        inpt = self.inputs[input_index]

        scl = inpt.scale // self.reference_data.scale
        slices = input.find_random_scene(
            self.samples[scene_index][1 + input_index],
            self.rng,
            multiple=16 // inpt.scale,
            window_size=self.window_size // scl,
            rqi_thresh=self.quality_threshold
        )

        if slices is None:
            new_index = self.rng.integers(0, len(self))
            return self[new_index]

        slices = tuple((index * scl for index in slices))


        xs = []
        ys = []

        if self.augment:
            ang = -180 + 360 * self.rng.random()
            flip = self.rng.random() > 0.5
        else:
            ang = None
            flip = False

        x, y = self.load_sample(
            self.samples[scene_index],
            slices,
            self.window_size,
            rotate=ang,
            flip=flip
        )

        has_input = any((x[inpt.name] is not None for inpt in self.inputs))
        has_output = any (
            (y[target.name] is not None for target in self.reference_data.targets)
        )

        if not has_input or not has_output:
            new_index = self.rng.integers(0, len(self))
            return self[new_index]

        return x, y


class SequenceDataset(SingleStepDataset):
    """
    Dataset class for temporal merging of satellite observations.
    """
    def __init__(
        self,
        folder,
        reference_data="mrms",
        inputs=None,
        sample_rate=2,
        normalize=True,
        window_size=256,
        sequence_length=32,
        start_time=None,
        end_time=None,
        quality_threshold=0.8,
        missing_value_policy="masked",
        augment=True,
        forecast=0,
        validation=False,
        time_step=None
    ):
        """
        Args:
            folder: The path to the training data.
            sample_rate: Rate for oversampling of training scenes.
            normalize: Whether to normalize the data.
            window_size: The size of the input data.
            sequence_length: The length of the training sequences.
            start_time: Optional start time to limit the samples.
            end_time: Optional end time to limit the available samples.
            augment: Whether to apply random transformations to the training
                inputs.
            augment: Whether to apply random flipping to the data.
            forecast: The number of samples in the sequence without input
                observations.
            validation: If 'True' sampling will reproduce identical scenes.
        """
        super().__init__(
            folder,
            reference_data=reference_data,
            inputs=inputs,
            sample_rate=sample_rate,
            normalize=normalize,
            window_size=window_size,
            start_time=start_time,
            missing_value_policy=missing_value_policy,
            end_time=end_time,
            quality_threshold=quality_threshold,
            augment=augment,
            validation=validation
        )

        self.sequence_length = sequence_length

        # Find samples with a series of consecutive outputs.
        times = self.keys
        deltas = times[sequence_length:] - times[:-sequence_length]
        if time_step is None:
            time_step = deltas.min()
        self.sequence_starts = np.where(
            deltas.astype("timedelta64[s]")
            <= sequence_length * time_step
        )[0]
        self.forecast = forecast


    def __len__(self):
        """Number of samples in an epoch."""
        return len(self.sequence_starts) * self.sample_rate


    def __getitem__(self, index):
        """Return training sample."""
        index = index // self.sample_rate

        # We load a larger window when input is rotated to avoid
        # missing values.
        if self.augment:
            window_size = int(1.42 * self.window_size)
            rem = window_size % self.max_scale
            if rem != 0:
                window_size += self.max_scale - rem
            ang = -180 + 360 * self.rng.random()
            flip = self.rng.random() > 0.5
        else:
            window_size = self.window_size
            ang = None
            flip = False

        # Find valid input range for last sample in sequence
        index = self.sequence_starts[index] + self.sequence_length
        slices = reference.find_random_scene(
            self.reference_data,
            self.samples[index][0],
            self.rng,
            multiple=4,
            window_size=window_size,
            qi_thresh=self.quality_threshold
        )

        x = []
        y = []
        start_index = self.sequence_starts[index]
        for i in range(self.sequence_length):
            index = start_index + i
            x_i, y_i = self.load_sample(
                self.samples[index],
                slices,
                self.window_size,
                rotate=ang,
                flip=flip
            )
            if i > self.sequence_length - self.forecast - 1:
                x.append({})
            else:
                x.append(x_i)
            y.append(y_i)
        return x, y


def plot_sample(x, y):
    """
    Plot input and output from a sample.

    Args:
        x: The training input.
        y: The training output.
    """
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]

    n_steps = len(x)

    for i in range(n_steps):
        f, axs = plt.subplots(1, 2, figsize=(10, 5))

        ax = axs[0]
        ax.imshow(x[i]["geo"][-1])

        ax = axs[1]
        ax.imshow(y[i])


def plot_date_distribution(
        path,
        keys=None,
        show_sensors=False,
        ax=None
):
    """
    Plot the number of training input samples per day.

    Args:
        path: Path to the directory that contains the training data.
        keys: A list file prefixes to look for, e.g. ['geo', 'visir'] to
            only list geostationary and AVHRR inputs.
        show_sensors: Whether to show different microwave sensors separately.
        ax: A matplotlib Axes object to use to draw the results.
    """
    if keys is None:
        keys = ["seviri", "mw", "radar", "visir"]
    if not isinstance(keys, list):
        keys = list(keys)

    times = {}
    time_min = None
    time_max = None
    for key in keys:

        files = list(Path(path).glob(f"**/{key}*.nc"))
        times_k = [
            datetime.strptime(name.name[len(key) + 1 : -3], "%Y%m%d_%H_%M")
            for name in files
        ]

        if key == "mw" and show_sensors:
            sensors = {}
            for time_k, filename in zip(times_k, files):
                with xr.open_dataset(filename) as data:
                    satellite = data.attrs["satellite"]
                    sensor = data.attrs["sensor"]
                    sensor = f"{sensor}"
                    sensor_times = sensors.setdefault(sensor, [])
                    sensor_times.append(time_k)

            for sensor in sensors:
                times_s = sensors[sensor]
                times_k = xr.DataArray(times_s)
                times[sensor] = times_k
                t_min = times_k.min()
                time_min = min(time_min, t_min) if time_min is not None else t_min
                t_max = times_k.max()
                time_max = max(time_max, t_max) if time_max is not None else t_max
        else:
            times_k = xr.DataArray(times_k)
            t_min = times_k.min()
            time_min = min(time_min, t_min) if time_min is not None else t_min
            t_max = times_k.max()
            time_max = max(time_max, t_max) if time_max is not None else t_max

            times[key] = times_k

    if ax is None:
        figure = plt.figure(figsize=(6, 4))
        ax = figure.add_subplot(1, 1, 1)

    bins = np.arange(
        time_min.astype("datetime64[D]").data,
        (time_max + np.timedelta64(2, "D")).astype("datetime64[D]").data,
        dtype="datetime64[D]",
    )
    x = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])

    for key in times:
        y, _ = np.histogram(times[key], bins=bins)
        ax.plot(x, y, label=key)

    ax.legend()
    return ax


def make_wave_x(size, w, theta, t=0.0):
    """
    Create a 2D image of a wave with given angular velocity and
    phase shift propagating along the last image dimension.
    """
    x = np.linspace(0, 2 * np.pi, size)
    x, y = np.meshgrid(x, x)
    return np.sin(w * x + theta + 2 * np.pi * t / 20.0)

def make_wave_y(size, w, theta, t=0.0):
    """
    Create a 2D image of a wave with given angular velocity and
    phase shift propagating along the last image dimension.
    """
    x = np.linspace(0, 2 * np.pi, size)
    x, y = np.meshgrid(x, x)
    return np.sin(w * y + theta + 2 * np.pi * t / 20.0)


class StreamData:
    """
    A synthetic dataset that requires the network to merge information
    from different streams.
    """
    def __init__(
            self,
            size=(128, 128),
            availability=(0.1, 0.1, 0.1),
            n_samples=5_000,
            sequence_length=1
    ):
        self.size = size
        self.init_function(0)
        if isinstance(availability, float):
            availability = [availability] * 3
        self.availability = availability
        self.n_samples = n_samples
        self.sequence_length = sequence_length


    def init_function(self, w_id):
        """
        Worker initialization function for multi-process data generation.
        Seeds the workers random generator.

        Args:
            w_id: Id of the worker.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples


    def make_sample(self, w_x, theta_x, w_y, theta_y, t=0.0):

        base_size = self.size[0]
        med_size = base_size // 2
        small_size = med_size // 2

        y_1 = make_wave_x(base_size, w_x, theta_x, t=t)
        w = self.rng.uniform(1, 3)
        theta = self.rng.uniform(0, np.pi)
        y_2 = make_wave_y(base_size, w_y, theta_y, t=t)

        y = y_1 + y_2

        x_visir = y_1 + 0.3 * self.rng.normal(size=y_1.shape)
        x_visir = np.tile(x_visir[np.newaxis], (5, 1, 1))
        r = self.rng.random()
        if r > self.availability[0]:
            x_visir = None

        y_2 = y_2[::2, ::2]
        x_geo = y_2 + 0.3 * self.rng.normal(size=y_2.shape)
        x_geo = np.tile(x_geo[np.newaxis], (11, 1, 1))

        r = self.rng.random()
        if r > self.availability[1]:
            x_geo = None

        y_2 = y_2[::2, ::2]
        x_mw_90 = np.tile(y_2[np.newaxis], (2, 1, 1))
        x_mw_160 = np.tile(y_2[np.newaxis], (2, 1, 1))
        y_1 = y_1[::4, ::4]
        x_mw_183 = np.tile(y_1[np.newaxis], (5, 1, 1))
        for i in range(2):
            i = self.rng.integers(0, 4)
            x_mw_183[i] = -3#self.rng.normal(size=x_mw_183.shape[1:])

        r = self.rng.random()
        if r > self.availability[2]:
            x_mw_90 = None
            x_mw_160 = None
            x_mw_183 = None

        def to_tensor(x):
            """
            Convert data to tensor or do nothing if it is None.
            """
            if x is not None:
                return torch.as_tensor(x).to(torch.float32)
            return x

        x = {
            "visir": to_tensor(x_visir),
            "geo": to_tensor(x_geo),
            "mw_90": to_tensor(x_mw_90),
            "mw_160": to_tensor(x_mw_160),
            "mw_183": to_tensor(x_mw_183)
        }
        return x, y


    def __getitem__(self, index):

        w_x = self.rng.uniform(1, 5)
        theta_x = self.rng.uniform(0, np.pi)
        w_y = self.rng.uniform(1, 5)
        theta_y = self.rng.uniform(0, np.pi)
        v = self.rng.uniform(-1.5, 1.5)
        if self.rng.random() > 0.5:
            v *= -1

        if self.sequence_length == 1:
            return self.make_sample(w_x, theta_x, w_y, theta_y)

        x = []
        y = []
        for i in range(self.sequence_length):
            x_i, y_i = self.make_sample(w_x, theta_x, w_y, theta_y, t=v*i)
            x.append(x_i)
            y.append(y_i)
        return x, y




class TestDataset:
    """
    A synthetic dataset that calculates a running average over the inputs
    thus forcing the network to learn to combine information across time
    steps. The purpose of this dataset is to test the sequence model
    architectures.
    """
    def __init__(
            self,
            sequence_length=1,
            size=(128, 128)
    ):
        """
        Args:
            sequence_length: The length of input sequences.
            size: The size of the inputs.
        """
        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.sequence_length = sequence_length

    def __len__(self):
        """Number of samples in dataset."""
        return 10_000

    def init_function(self, w_id):
        """
        Worker initialization function for multi-process data generation.
        Seeds the workers random generator.

        Args:
            w_id: Id of the worker.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):
        """
        Generate a training sample.

        Args:
            index: Not used.
        """
        x_visir = [
            self.rng.uniform(-1, 1) * np.ones((5,) + self.size, dtype=np.float32)
            for i in range(self.sequence_length)
        ]

        size = tuple([s // 2 for s in self.size])
        x_geo = [
            self.rng.uniform(-1, 1) * np.ones((11,) + size, dtype=np.float32)
            for i in range(self.sequence_length)
        ]

        size = tuple([s // 4 for s in self.size])
        x_mw_90 = [
            self.rng.uniform(-1, 1) * np.ones((2,) + size, dtype=np.float32)
            for i in range(self.sequence_length)
        ]
        x_mw_160 = [
            self.rng.uniform(-1, 1) * np.ones((2,) + size, dtype=np.float32)
            for i in range(self.sequence_length)
        ]
        x_mw_183 = [
            self.rng.uniform(-1, 1) * np.ones((5,) + size, dtype=np.float32)
            for i in range(self.sequence_length)
        ]

        y = np.stack(x_visir, axis=0)[:, 0]
        y[1:] += y[:-1]
        y[1:] *= 0.5

        y = [torch.as_tensor(y_i, dtype=torch.float32) for y_i in y]

        xs = []
        for visir, geo, mw_90, mw_160, mw_183 in zip(x_visir, x_geo, x_mw_90, x_mw_160, x_mw_183):
            visir = (
                torch.as_tensor(visir, dtype=torch.float32) +
                torch.as_tensor(self.rng.uniform(-0.05, 0.05, size=visir.shape),
                             dtype=torch.float32)
            )
            geo = (
                torch.as_tensor(geo, dtype=torch.float32) +
                torch.as_tensor(self.rng.uniform(-0.05, 0.05, size=geo.shape),
                             dtype=torch.float32)
            )
            geo[:5] = visir[..., ::2, ::2]
            mw_90 = (
                torch.as_tensor(mw_90, dtype=torch.float32) +
                torch.as_tensor(self.rng.uniform(-0.05, 0.05, size=mw_90.shape),
                             dtype=torch.float32)
            )
            mw_160 = (
                torch.as_tensor(mw_160, dtype=torch.float32) +
                torch.as_tensor(self.rng.uniform(-0.05, 0.05, size=mw_160.shape),
                             dtype=torch.float32)
            )
            mw_183 = (
                torch.as_tensor(mw_183, dtype=torch.float32) +
                torch.as_tensor(self.rng.uniform(-0.05, 0.05, size=mw_183.shape),
                             dtype=torch.float32)
            )

            xs.append({
                "visir": visir,
                "geo": geo,
                "mw_90": mw_90,
                "mw_160": mw_160,
                "mw_183": mw_183
            })
        return xs, y


def random_spectral_field(rng, n, lower, upper, energy):
    """
    Create a 2D random field with a banded spectral signature.

    Args:
        n: The size of the field.
        lower: The lower wavelength bound of the spectral band
            with non-zero energy.
        upper: The upper wavelength bound of the spectral band
            with non-zero energy.

    Return:
        A 2D array containing the random field.
    """
    v = np.zeros((n, n), dtype=np.float32)
    l = np.arange(n)
    l = 1.0 / np.sqrt(l.reshape(-1, 1) ** 2 + l.reshape(1, -1) ** 2)

    mask = (l >= lower) * (l < upper)
    v[mask] = rng.uniform(-1, 1, size=mask.sum()).astype(np.float32)
    e = sum(v[mask] ** 2)
    v[mask] *= np.sqrt(energy / e)
    v = v * np.sqrt(n ** 2)
    return v

def normalize_field(v, energy):
    n = v.shape[0]
    e = sum(v ** 2)
    nothing = e < 1e-9
    e[nothing] = 1.0
    v[nothing] = 0.0
    return v * np.sqrt(energy / e) * np.sqrt(n ** 2)



class SuperpositionDataset:
    """
    Synthetic dataset mapping band-filtered and corrupted views of a random
    field to the full random field.
    """
    def __init__(
            self,
            size,
            n_samples=1000,
            availability=None,
            sparse=False,
            n_steps=1,
            snr=0.0,
            composition="sum"
    ):
        """
        Args:
            size: The size of the random field.
            n_samples: The number of training samples in the dataset.
            availability: The availability of the three inputs.
            sparse: Whether the input should be returned as packed
                tensor.
            n_steps: The number of frames per sample. If this is larger
                than one, each sample is a sequence of inputs and corresponding
                outputs.
            snr: The signal to noise ratio (SNR) determining the strength of
                the noise in the input observations.
        """
        self.size = size
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.snr = snr
        if availability is None:
            availability = [0.3, 0.3, 0.3]
        elif isinstance(availability, float):
            availability = [availability] * 3
        self.availability = availability
        self.sparse = sparse
        self.composition = composition
        self.init_rng()

    def init_rng(self, w_id=0):
        """
        Initialize random number generator.

        Args:
            w_id: The worker ID which of the worker process..
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        return self.make_sample(n_steps=self.n_steps)

    def make_sample(self, n_steps=1):
        """
        Creates a pair of input and output fields.
        """
        xs = []
        ys = []

        low_s = random_spectral_field(self.rng, self.size, 0.3, 1.0, 1.0)
        med_s = random_spectral_field(self.rng, self.size, 0.2, 0.3, 1.0)
        hi_s = random_spectral_field(self.rng, self.size, 0.05, 0.2, 1.0)
        low_e = random_spectral_field(self.rng, self.size, 0.3, 1.0, 1.0)
        med_e = random_spectral_field(self.rng, self.size, 0.2, 0.3, 1.0)
        hi_e = random_spectral_field(self.rng, self.size, 0.05, 0.2, 1.0)

        t =  random_spectral_field(self.rng, self.size, 0.1, 0.2, 1.0)

        for i in range(n_steps):

            l = i / n_steps
            r = 1.0 - l
            low = fft.idctn(l * low_s + r * low_e, norm="ortho")
            med = fft.idctn(l * med_s + r * med_e, norm="ortho")
            hi = fft.idctn(l * hi_s + r * hi_e, norm="ortho")

            if self.composition == "sum":
                y = low + med + hi
            else:
                y = np.abs(low) * np.abs(med) + np.abs(med) * np.abs(hi)

            x_visir = (
                hi[None] +
                self.snr * self.rng.normal(size=(5,) + (self.size,) * 2)
            )
            x_geo = (
                med[None, ::2, ::2] +
                self.snr * self.rng.normal(size=(11,) + (self.size // 2,) * 2)
            )
            x_mw_90 = (
                low[None, ::4, ::4] +
                self.snr * self.rng.normal(size=(2,) + (self.size // 4,) * 2)
            )
            x_mw_160 = (
                low[None, ::4, ::4] +
                self.snr * self.rng.normal(size=(2,) + (self.size // 4,) * 2)
            )
            x_mw_183 = (
                low[None, ::4, ::4] +
                self.snr * self.rng.normal(size=(5,) + (self.size // 4,) * 2)
            )

            if (self.rng.random() > self.availability[0]):
                if self.sparse:
                    x_visir = None
                else:
                    x_visir[:] = -3
            if (self.rng.random() > self.availability[1]):
                if self.sparse:
                    x_geo = None
                else:
                    x_geo[:] = -3
            if (self.rng.random() > self.availability[2]):
                if self.sparse:
                    x_mw_90 = None
                    x_mw_160 = None
                    x_mw_183 = None
                else:
                    x_mw_90[:] = -3
                    x_mw_160[:] = -3
                    x_mw_183[:] = -3

            if x_visir is not None:
                x_visir = x_visir.astype(np.float32)
            if x_geo is not None:
                x_geo = x_geo.astype(np.float32)
            if x_mw_90 is not None:
                x_mw_90 = x_mw_90.astype(np.float32)
            if x_mw_160 is not None:
                x_mw_160 = x_mw_160.astype(np.float32)
            if x_mw_183 is not None:
                x_mw_183 = x_mw_183.astype(np.float32)

            x = {
                "visir": x_visir,
                "geo": x_geo,
                "mw_90": x_mw_90,
                "mw_160": x_mw_160,
                "mw_183": x_mw_183
            }
            xs.append(x)
            ys.append(y)
        if n_steps == 1:
            return xs[0], ys[0]
        return xs, ys

    def plot_sample(self, x, y):
        norm = Normalize(-3, 3)
        f = plt.figure(figsize=(22, 5))
        axs = np.array([f.add_subplot(1, 4, i + 1) for i in range(4)])

        if self.n_steps == 1:
            ind =  -1
            ax = axs[0]
            ax.imshow(x["visir"][0], norm=norm)
            ax.set_title("(a) VISIR", loc="left")

            ax = axs[1]
            ax.imshow(x["geo"][0], norm=norm)
            ax.set_title("(b) GEO", loc="left")

            ax = axs[2]
            ax.imshow(x["mw_183"][1], norm=norm)
            ax.set_title("(c) MW", loc="left")

            ax = axs[3]
            ax.set_title("(d) Output", loc="left")
            ax.imshow(y, norm=norm)

            return f
        else:
            def draw_frame(index):
                x_i = x[index]
                y_i = y[index]

                ax = axs[0]
                ax.imshow(x_i["visir"][0], norm=norm)
                ax.set_title("(a) VISIR", loc="left")

                ax = axs[1]
                ax.imshow(x_i["geo"][0], norm=norm)
                ax.set_title("(b) GEO", loc="left")

                ax = axs[2]
                ax.imshow(x_i["mw_183"][1], norm=norm)
                ax.set_title("(c) MW", loc="left")

                ax = axs[3]
                ax.set_title("(d) Output", loc="left")
                ax.imshow(y_i, norm=norm)

            return FuncAnimation(
                f, draw_frame, range(len(x))
            )

