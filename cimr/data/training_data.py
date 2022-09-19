"""
cimr.data.training_data
=======================

Interface classes for loading the CIMR training data.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from quantnn.normalizer import MinMaxNormalizer
import torch
from torch import nn
import xarray as xr
import pandas as pd

from cimr.areas import NORDIC_2
from cimr.data.mhs import MHS
from cimr.utils import MISSING

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


def load_geo_obs(sample, dataset, normalize=True, rng=None):
    """
    Loads geostationary observations from dataset.
    """
    data = []
    for i in range(11):
        name = f"geo_{i + 1:02}"
        data.append(dataset[name].data)

    data = np.stack(data, axis=0)
    if normalize:
        data = NORMALIZER_GEO(data, rng=rng)
    sample["geo"] = torch.tensor(data)


def load_visir_obs(sample, dataset, normalize=True, rng=None):
    """
    Loads VIS/IR observations from dataset.
    """
    data = []
    for i in range(5):
        name = f"visir_{i + 1:02}"
        data.append(dataset[name].data)

    data = np.stack(data, axis=0)
    if normalize:
        data = NORMALIZER_VISIR(data, rng=rng)
    sample["visir"] = torch.tensor(data)


def load_microwave_obs(sample, dataset, normalize=True, rng=None):
    """
    Loads microwave observations from dataset.
    """
    shape = (dataset.y.size, dataset.x.size)
    if "mw_90" in dataset:
        x = np.transpose(dataset.mw_90.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_90(x, rng=rng)
        sample["mw_90"] = torch.tensor(x)
    else:
        sample["mw_90"] = MISSING * torch.ones((2,) + shape)

    if "mw_160" in dataset:
        x = np.transpose(dataset.mw_160.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_160(x, rng=rng)
        sample["mw_160"] = torch.tensor(x)
    else:
        sample["mw_160"] = MISSING * torch.ones((2,) + shape)

    if "mw_183" in dataset:
        x = np.transpose(dataset.mw_183.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_183(x, rng=rng)
        sample["mw_183"] = torch.tensor(x)
    else:
        sample["mw_160"] = MISSING * torch.ones((5,) + shape)


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


def get_date(filename):
    """
    Extract date from a training data filename.

    Args:
        filename: The name of the file containing

    Return:
        Numpy datetime64 object containing the time corresponding to
        the training sample.
    """
    _, yearmonthday, hour, minute = filename.stem.split("_")
    year = yearmonthday[:4]
    month = yearmonthday[4:6]
    day = yearmonthday[6:]
    return np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")


class CIMRDataset:
    """
    Dataset class for the CIMR training data.

    Implements the PyTorch Dataset interface.
    """
    def __init__(
        self,
        folder,
        sample_rate=1,
        sequence_length=1,
        normalize=True,
        window_size=128,
        start_time=None,
        end_time=None,
        quality_threshold=0.8,
    ):
        """
        Args:
            folder: The root folder containing the training data.
            sample_rate: How often each scene should be sampled per epoch.
            normalize: Whether inputs should be normalized.
            window_size: Size of the training scenes.
            start_time: Start time of time interval to which to restrict
                training data.
            end_time: End time of a time interval to which to restrict the
                training data.
            quality_threshold: Threshold for radar quality index used to mask
                the radar measurements.
        """
        self.folder = Path(folder)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.quality_threshold = quality_threshold

        radar_files = sorted(list((self.folder / "radar").glob("radar*.nc")))
        times = np.array(list(map(get_date, radar_files)))
        if start_time is not None and end_time is not None:
            indices = (times >= start_time) * (times < end_time)
            times = times[indices]
            radar_files = [radar_files[i] for i in np.where(indices)[0]]

        self.samples = {
            time: SampleRecord(b_file) for time, b_file in zip(times, radar_files)
        }

        geo_files = sorted(list((self.folder / "geo").glob("*.nc")))
        times = np.array(list(map(get_date, geo_files)))
        for time, geo_file in zip(times, geo_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.geo = geo_file

        mw_files = sorted(list((self.folder / "microwave").glob("*.nc")))
        times = np.array(list(map(get_date, mw_files)))
        for time, mw_file in zip(times, mw_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.mw = mw_file

        visir_files = sorted(list((self.folder / "visir").glob("*.nc")))
        times = np.array(list(map(get_date, visir_files)))
        for time, visir_file in zip(times, visir_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.visir = visir_file

        self.keys = np.array(list(self.samples.keys()))

        self.window_size = window_size

        # Determine valid start point for input samples.
        self.sequence_length = sequence_length
        times = np.array(list(self.samples.keys()))
        deltas = times[self.sequence_length :] - times[: -self.sequence_length]
        starts = np.where(
            deltas.astype("timedelta64[s]")
            <= np.timedelta64(self.sequence_length * 15 * 60, "s")
        )[0]
        self.sequence_starts = starts

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
        """Number of samples in dataset."""
        return len(self.sequence_starts) * self.sample_rate

    def load_sample(self, index, slices=None):
        """
        Load training sample.

        Args:
            index: Time stamp identifying the input sample.
            slices: Optional row and column slices determining the
                window to extract from each scene.
        """
        key = self.keys[index]
        with xr.open_dataset(self.samples[key].radar) as data:

            y = data.dbz.data.copy()
            y[data.qi < self.quality_threshold] = np.nan

            if slices is None:

                found = False
                while not found:
                    n_rows, n_cols = y.shape
                    i_start = self.rng.integers(0, (n_rows - self.window_size) // 4)
                    i_end = i_start + self.window_size // 4
                    j_start = self.rng.integers(0, (n_cols - self.window_size) // 4)
                    j_end = j_start + self.window_size // 4

                    row_slice = slice(4 * i_start, 4 * i_end)
                    col_slice = slice(4 * j_start, 4 * j_end)

                    y_s = y[row_slice, col_slice]

                    if (y_s >= -100).mean() > 0.2:
                        found = True

            else:
                i_start, i_end, j_start, j_end = slices
                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)
                y_s = y[row_slice, col_slice]

        y_s = np.nan_to_num(y_s, nan=-100)
        y = torch.tensor(y_s, dtype=torch.float)

        x = {}

        # GEO data
        if self.samples[key].geo is not None:

            with xr.open_dataset(self.samples[key].geo) as data:
                row_slice = slice(i_start * 2, i_end * 2)
                col_slice = slice(j_start * 2, j_end * 2)
                load_geo_obs(
                    x, data[{"y": row_slice, "x": col_slice}], normalize=self.normalize,
                    rng=self.rng
                )
        else:
            x["geo"] = torch.tensor(
                NORMALIZER_GEO(
                    np.nan * np.ones((11,) + (self.window_size // 2,) * 2),
                    rng=self.rng
                ),
                dtype=torch.float,
            )

        # VISIR data
        if self.samples[key].visir is not None:
            with xr.open_dataset(self.samples[key].visir) as data:
                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)
                load_visir_obs(
                    x, data[{"y": row_slice, "x": col_slice}], normalize=self.normalize,
                    rng=self.rng
                )
        else:
            x["visir"] = torch.tensor(
                NORMALIZER_VISIR(
                    np.nan * np.ones((5,) + (self.window_size,) * 2),
                    rng=self.rng
                ),
                dtype=torch.float
            )

        # Microwave data
        if self.samples[key].mw is not None:
            with xr.open_dataset(self.samples[key].mw) as data:
                row_slice = slice(i_start, i_end)
                col_slice = slice(j_start, j_end)
                load_microwave_obs(
                    x, data[{"y": row_slice, "x": col_slice}], normalize=self.normalize,
                    rng=self.rng
                )
        else:
            x["mw_90"] = torch.tensor(
                NORMALIZER_MW_90(
                    np.nan * np.ones((2,) + (self.window_size // 4,) * 2),
                    rng=self.rng
                ),
                dtype=torch.float,
            )
            x["mw_160"] = torch.tensor(
                NORMALIZER_MW_160(
                    np.nan * np.ones((2,) + (self.window_size // 4,) * 2),
                    rng=self.rng
                ),
                dtype=torch.float,
            )
            x["mw_183"] = torch.tensor(
                NORMALIZER_MW_183(
                    np.nan * np.ones((5,) + (self.window_size // 4,) * 2),
                    rng=self.rng
                ),
                dtype=torch.float,
            )

        return x, y

    def __getitem__(self, index):
        """Return ith training sample."""

        scene_index = index // self.sample_rate
        key = self.keys[self.sequence_starts[scene_index]]

        with xr.open_dataset(self.samples[key].radar) as data:

            y = data.dbz.data.copy()
            y[data.qi < self.quality_threshold] = np.nan

            found = False
            while not found:

                n_rows, n_cols = y.shape

                i_start = self.rng.integers(0, (n_rows - self.window_size) // 4)
                i_end = i_start + self.window_size // 4
                j_start = self.rng.integers(0, (n_cols - self.window_size) // 4)
                j_end = j_start + self.window_size // 4

                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)

                y_s = y[row_slice, col_slice]

                if (y_s >= -100).mean() > 0.2:
                    found = True

        slices = (i_start, i_end, j_start, j_end)

        xs = []
        ys = []

        # If not in sequence mode return data directly.
        if self.sequence_length == 1:
            return self.load_sample(scene_index, slices=slices)

        # Otherwise collect samples in list.
        for i in range(self.sequence_length):
            x, y = self.load_sample(
                self.sequence_starts[scene_index] + i, slices=slices
            )
            xs.append(x)
            ys.append(y)
        return xs, ys

    def plot(self, key):

        crs = NORDIC_2.to_cartopy_crs()
        extent = NORDIC_2.area_extent
        extent = (extent[0], extent[2], extent[1], extent[3])

        f = plt.figure(figsize=(10, 12))
        gs = GridSpec(2, 2)
        ax = axs[0, 0]
        dbz = radar_data.dbz.data.copy()
        dbz[radar_data.qi < self.quality_threshold] = np.nan
        img = ax.imshow(dbz, extent=extent, vmin=-20, vmax=20)
        ax.coastlines(color="grey")

        ax = axs[0, 1]
        if sample.geo is not None:
            geo_data = xr.load_dataset(sample.geo)
            ax.imshow(geo_data.geo_10, extent=extent)
            ax.coastlines(color="grey")
        else:
            img = np.nan * np.ones((2, 2))
            img = ax.imshow(img, extent=extent)
            ax.coastlines(color="grey")

        ax = axs[1, 0]
        if sample.visir is not None:
            visir_data = xr.load_dataset(sample.visir)
            img = ax.imshow(visir_data.visir_05, extent=extent)
            ax.coastlines(color="grey")
        else:
            img = np.nan * np.ones((2, 2))
            img = ax.imshow(img, extent=extent)
            ax.coastlines(color="grey")

        ax = axs[1, 1]
        if sample.mw is not None:
            mw_data = xr.load_dataset(sample.mw)
            img = ax.imshow(mw_data.mw_183.data[..., -1], extent=extent)
            ax.coastlines(color="grey")
        else:
            img = np.nan * np.ones((2, 2))
            img = ax.imshow(img, extent=extent)
            ax.coastlines(color="grey")

            return [img]

        return f, axs

    def make_animator(self, start_time, end_time):
        indices = (self.keys >= start_time) * (self.keys <= end_time)
        keys = self.keys[indices]

        crs = NORDIC_2.to_cartopy_crs()
        extent = NORDIC_2.area_extent
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
            dbz[radar_data.qi < self.quality_threshold] = np.nan
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
            y = data.dbz.data.copy()
            y[data.qi < self.quality_threshold] = np.nan

        y = torch.tensor(y, dtype=torch.float)
        shape = y.shape

        x = {}

        # VISIR data
        if self.samples[key].visir is not None:
            with xr.open_dataset(self.samples[key].visir) as data:
                load_visir_obs(x, data, normalize=self.normalize, rng=self.rng)
        else:
            x["visir"] = torch.tensor(
                NORMALIZER_VISIR(np.nan * np.ones((5,) + shape), rng=self.rng),
                dtype=torch.float
            )

        shape = tuple([n // 2 for n in y.shape])
        if self.samples[key].geo is not None:
            with xr.open_dataset(self.samples[key].geo) as data:
                load_geo_obs(x, data, normalize=self.normalize, rng=self.rng)
        else:
            x["geo"] = torch.tensor(
                NORMALIZER_GEO(np.nan * np.ones((11,) + shape), rng=self.rng),
                dtype=torch.float
            )

        shape = tuple([n // 4 for n in y.shape])
        # Microwave data
        if self.samples[key].mw is not None:
            with xr.open_dataset(self.samples[key].mw) as data:
                load_microwave_obs(x, data, normalize=self.normalize, rng=self.rng
                )
        else:
            x["mw_90"] = torch.tensor(
                NORMALIZER_MW_90(np.nan * np.ones((2,) + shape), rng=self.rng),
                dtype=torch.float,
            )
            x["mw_160"] = torch.tensor(
                NORMALIZER_MW_160(np.nan * np.ones((2,) + shape), rng=self.rng),
                dtype=torch.float,
            )
            x["mw_183"] = torch.tensor(
                NORMALIZER_MW_183(MISSING * np.ones((5,) + shape), rng=self.rng),
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

        input_visir = nn.functional.pad(input_visir, padding, "replicate")
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
        input_geo = nn.functional.pad(input_geo, padding, "replicate")
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

        input_mw_90 = nn.functional.pad(input_mw_90, padding, "replicate")
        x["mw_90"] = input_mw_90
        input_mw_160 = nn.functional.pad(input_mw_160, padding, "replicate")
        x["mw_160"] = input_mw_160
        input_mw_183 = nn.functional.pad(input_mw_183, padding, "replicate")
        x["mw_183"] = input_mw_183

        return x, slice_y, slice_x

    def full_range(self, start_time=None, end_time=None):

        indices = np.ones(self.keys.size, dtype=np.bool)
        if start_time is not None:
            indices = indices * (self.keys >= start_time)
        if end_time is not None:
            indices = indices * (self.keys <= end_time)
        keys = sorted(self.keys[indices])

        for key in keys:
            x, y = self.load_full_data(key)
            x, slice_y, slice_x = self.pad_input(x)

            for k, v in x.items():
                x[k] = v.unsqueeze(0)

            yield x, y, slice_y, slice_x, key


class CIMRSequenceDataset(CIMRDataset):
    """
    Dataset class for the CIMR training data.
    """

    def __init__(
        self,
        folder,
        sample_rate=4,
        normalize=True,
        window_size=128,
        sequence_length=32,
        start_time=None,
        end_time=None,
        quality_threshold=0.8,
    ):
        super().__init__(
            folder,
            sample_rate=sample_rate,
            normalize=normalize,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time,
            quality_threshold=quality_threshold,
        )

        self.sequence_length = sequence_length
        times = np.array(list(self.samples.keys()))
        deltas = times[self.sequence_length :] - times[: -self.sequence_length]
        starts = np.where(
            deltas.astype("timedelta64[s]")
            <= np.timedelta64(self.sequence_length * 15 * 60, "s")
        )[0]
        self.sequence_starts = starts

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, index):

        key = self.keys[self.sequence_starts[index]]
        with xr.open_dataset(self.samples[key].radar) as data:

            y = data.dbz.data.copy()
            y[data.qi < self.quality_threshold] = np.nan

            found = False
            while not found:

                n_rows, n_cols = y.shape

                i_start = self.rng.integers(0, (n_rows - self.window_size) // 4)
                i_end = i_start + self.window_size // 4
                j_start = self.rng.integers(0, (n_cols - self.window_size) // 4)
                j_end = j_start + self.window_size // 4

                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)

                y_s = y[row_slice, col_slice]

                if (y_s >= -100).mean() > 0.2:
                    found = True

        slices = (i_start, i_end, j_start, j_end)

        xs = []
        ys = []
        for i in range(self.sequence_length):
            x, y = self.load_sample(self.sequence_starts[index] + i, slices=slices)
            xs.append(x)
            ys.append(y)
        return xs, ys


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

        y = [torch.tensor(y_i, dtype=torch.float32) for y_i in y]

        xs = []
        for visir, geo, mw_90, mw_160, mw_183 in zip(x_visir, x_geo, x_mw_90, x_mw_160, x_mw_183):
            visir = (
                torch.tensor(visir, dtype=torch.float32) +
                torch.tensor(self.rng.uniform(-0.05, 0.05, size=visir.shape),
                             dtype=torch.float32)
            )
            geo = (
                torch.tensor(geo, dtype=torch.float32) +
                torch.tensor(self.rng.uniform(-0.05, 0.05, size=geo.shape),
                             dtype=torch.float32)
            )
            geo[:5] = visir[..., ::2, ::2]
            mw_90 = (
                torch.tensor(mw_90, dtype=torch.float32) +
                torch.tensor(self.rng.uniform(-0.05, 0.05, size=mw_90.shape),
                             dtype=torch.float32)
            )
            mw_160 = (
                torch.tensor(mw_160, dtype=torch.float32) +
                torch.tensor(self.rng.uniform(-0.05, 0.05, size=mw_160.shape),
                             dtype=torch.float32)
            )
            mw_183 = (
                torch.tensor(mw_183, dtype=torch.float32) +
                torch.tensor(self.rng.uniform(-0.05, 0.05, size=mw_183.shape),
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
