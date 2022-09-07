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
    0: (0.0, 100.0),
    1: (0.0, 100.0),
    2: (170, 320),
    3: (210, 310),
    4: (210, 310),
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


def load_geo_obs(sample, dataset, normalize=True):
    """
    Loads geostationary observations from dataset.
    """
    data = []
    for i in range(11):
        name = f"geo_{i + 1:02}"
        data.append(dataset[name].data)

    data = np.stack(data, axis=0)
    if normalize:
        data = NORMALIZER_GEO(data)
    sample["geo"] = torch.tensor(data)


def load_visir_obs(sample, dataset, normalize=True):
    """
    Loads VIS/IR observations from dataset.
    """
    data = []
    for i in range(5):
        name = f"visir_{i + 1:02}"
        data.append(dataset[name].data)

    data = np.stack(data, axis=0)
    if normalize:
        data = NORMALIZER_VISIR(data)
    sample["visir"] = torch.tensor(data)


def load_microwave_obs(sample, dataset, normalize=True):
    """
    Loads microwave observations from dataset.
    """
    shape = (dataset.y.size, dataset.x.size)
    if "mw_90" in dataset:
        x = np.transpose(dataset.mw_90.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_90(x)
        sample["mw_90"] = torch.tensor(x).to(torch.float)
    else:
        sample["mw_90"] = MISSING * torch.ones((2,) + shape).to(torch.float)

    if "mw_160" in dataset:
        x = np.transpose(dataset.mw_160.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_160(x)
        sample["mw_160"] = torch.tensor(x).to(torch.float)
    else:
        sample["mw_160"] = MISSING * torch.ones((2,) + shape).to(torch.float)

    if "mw_183" in dataset:
        x = np.transpose(dataset.mw_183.data, (2, 0, 1))
        if normalize:
            x = NORMALIZER_MW_183(x)
        sample["mw_183"] = torch.tensor(x).to(torch.float)
    else:
        sample["mw_160"] = MISSING * torch.ones((5,) + shape).to(torch.float)


@dataclass
class SampleRecord:
    """
    Record holding the paths of the files for single training
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
    """
    def __init__(self,
                 folder,
                 sample_rate=1,
                 sequence_length=1,
                 normalize=True,
                 window_size=128,
                 start_time=None,
                 end_time=None
    ):
        """
        Args:
            folder: The root folder containing the training data.
            sample_rate: How often each scene should be sampled per epoch.
            normalize: Whether inputs should be normalized.
            window_size: Size of the training scenes.
            start_time: Start time of time interval to which to restrict training
                data.
            end_time: End time of a time interval to which to restrict the training
                data.
        """
        self.folder = Path(folder)
        self.sample_rate = sample_rate
        self.normalize = normalize

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
        deltas = times[self.sequence_length:] - times[:-self.sequence_length]
        starts = np.where(
            deltas.astype("timedelta64[s]") <=
            np.timedelta64(self.sequence_length * 15 * 60, "s")
        )[0]
        self.sequence_starts = starts

    def __len__(self):
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
            y[data.qi < 0.8] = np.nan

            if slices is None:

                found = False
                while not found:
                    n_rows, n_cols = y.shape
                    i_start = np.random.randint(0, (n_rows - self.window_size) // 4)
                    i_end = i_start + self.window_size // 4
                    j_start = np.random.randint(0, (n_cols - self.window_size) // 4)
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
                    x,
                    data[{"y": row_slice, "x": col_slice}],
                    normalize=self.normalize
                )
        else:
            x["geo"] = torch.tensor(
                MISSING * np.ones((11,) + (self.window_size // 2,) * 2),
                dtype=torch.float,
            )

        # VISIR data
        if self.samples[key].visir is not None:
            with xr.open_dataset(self.samples[key].visir) as data:
                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)
                load_visir_obs(
                    x,
                    data[{"y": row_slice, "x": col_slice}],
                    normalize=self.normalize)
        else:
            x["visir"] = torch.tensor(
                MISSING * np.ones((5,) + (self.window_size,) * 2), dtype=torch.float
            )

        # Microwave data
        if self.samples[key].mw is not None:
            with xr.open_dataset(self.samples[key].mw) as data:
                row_slice = slice(i_start, i_end)
                col_slice = slice(j_start, j_end)
                load_microwave_obs(
                    x,
                    data[{"y": row_slice, "x": col_slice}],
                    normalize=self.normalize
                )
        else:
            x["mw_90"] = torch.tensor(
                MISSING * np.ones((2,) + (self.window_size // 4,) * 2),
                dtype=torch.float,
            )
            x["mw_160"] = torch.tensor(
                MISSING * np.ones((2,) + (self.window_size // 4,) * 2),
                dtype=torch.float,
            )
            x["mw_183"] = torch.tensor(
                MISSING * np.ones((5,) + (self.window_size // 4,) * 2),
                dtype=torch.float,
            )

        return x, y

    def __getitem__(self, index):

        scene_index = index // self.sample_rate
        key = self.keys[self.sequence_starts[scene_index]]

        with xr.open_dataset(self.samples[key].radar) as data:

            y = data.dbz.data.copy()
            y[data.qi < 0.8] = np.nan

            found = False
            while not found:

                n_rows, n_cols = y.shape

                i_start = (n_rows - self.window_size) // 8
                i_end = (n_rows + self.window_size) // 8
                j_start = (n_cols - self.window_size) // 8
                j_end = (n_cols + self.window_size) // 8
                found = True
                #i_start = np.random.randint(0, (n_rows - self.window_size) // 4)
                #i_end = i_start + self.window_size // 4
                #j_start = np.random.randint(0, (n_cols - self.window_size) // 4)
                #j_end = j_start + self.window_size // 4

                #row_slice = slice(4 * i_start, 4 * i_end)
                #col_slice = slice(4 * j_start, 4 * j_end)

                #y_s = y[row_slice, col_slice]

                #if (y_s >= -100).mean() > 0.2:
                #    found = True

        slices = (i_start, i_end, j_start, j_end)

        xs = []
        ys = []

        # If not in sequence mode return data directly.
        if self.sequence_length == 1:
            return self.load_sample(scene_index, slices=slices)

        # Otherwise collect samples in list.
        for i in range(self.sequence_length):
            x, y = self.load_sample(self.sequence_starts[scene_index] + i, slices=slices)
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
        dbz[radar_data.qi < 0.8] = np.nan
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
            dbz[radar_data.qi < 0.8] = np.nan
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

    def load_full_data(self, key, min_qi=0.8):
        with xr.open_dataset(self.samples[key].radar) as data:
            y = data.dbz.data.copy()
            y[data.qi < 0.8] = np.nan

        y = torch.tensor(y, dtype=torch.float)
        shape = y.shape

        x = {}
        # GEO data
        if self.samples[key].geo is not None:
            with xr.open_dataset(self.samples[key].geo) as data:
                xs = np.stack([data[f"channel_{i:02}"].data for i in range(1, 13)])
            x["geo"] = torch.tensor(NORMALIZER_GEO(xs), dtype=torch.float)
        else:
            x["geo"] = torch.tensor(
                MISSING * np.ones((12,) + tuple([s // 2 for s in shape])),
                dtype=torch.float,
            ) # VISIR data
        if self.samples[key].avhrr is not None:
            with xr.open_dataset(self.samples[key].avhrr) as data:
                xs = np.stack([data[f"channel_{i:01}"].data for i in range(1, 6)])
            x["avhrr"] = torch.tensor(NORMALIZER_VISIR(xs), dtype=torch.float)
        else:
            x["avhrr"] = torch.tensor(
                MISSING * np.ones((5,) + shape), dtype=torch.float
            )

        # MHS data
        if self.samples[key].mhs is not None:
            with xr.open_dataset(self.samples[key].mhs) as data:
                xs = np.stack([data[f"channel_{i:02}"].data for i in range(1, 6)])
            x["mhs"] = torch.tensor(NORMALIZER_MHS(xs), dtype=torch.float)
        else:
            x["mhs"] = torch.tensor(
                MISSING * np.ones((5,) + tuple([s // 4 for s in shape])),
                dtype=torch.float,
            )

        return x, y

    def pad_input(self, x, multiple=32):

        input_avhrr = x["avhrr"]
        input_geo = x["geo"]
        input_mhs = x["mhs"]

        shape = input_avhrr.shape[1:]

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

        slice_x = slice(padding_x_l, -padding_x_r)
        slice_y = slice(padding_y_l, -padding_y_r)

        input_avhrr = nn.functional.pad(input_avhrr, padding, "replicate")
        x["avhrr"] = input_avhrr

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

        input_mhs = nn.functional.pad(input_mhs, padding, "replicate")
        x["mhs"] = input_mhs

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

            x["geo"] = x["geo"].unsqueeze(0)
            x["avhrr"] = x["avhrr"].unsqueeze(0)
            x["mhs"] = x["mhs"].unsqueeze(0)


            yield x, y, slice_y, slice_x


class CIMRSequenceDataset(CIMRDataset):
    """
    Dataset class for the CIMR training data.
    """
    def __init__(self,
                 folder,
                 sample_rate=4,
                 normalize=True,
                 window_size=128,
                 sequence_length=32,
                 start_time=None,
                 end_time=None
    ):
        super().__init__(
            folder,
            sample_rate=sample_rate,
            normalize=normalize,
            window_size=window_size,
            start_time=start_time,
            end_time=end_time
        )

        self.sequence_length = sequence_length
        times = np.array(list(self.samples.keys()))
        deltas = times[self.sequence_length:] - times[:-self.sequence_length]
        starts = np.where(
            deltas.astype("timedelta64[s]") <= np.timedelta64(self.sequence_length * 15 * 60, "s")
        )[0]
        self.sequence_starts = starts


    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, index):

        key = self.keys[self.sequence_starts[index]]
        with xr.open_dataset(self.samples[key].radar) as data:

            y = data.dbz.data.copy()
            y[data.qi < 0.8] = np.nan

            found = False
            while not found:

                n_rows, n_cols = y.shape

                i_start = np.random.randint(0, (n_rows - self.window_size) // 4)
                i_end = i_start + self.window_size // 4
                j_start = np.random.randint(0, (n_cols - self.window_size) // 4)
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


def plot_date_distribution(path, keys=None):

    if keys is None:
        keys = ["seviri", "mhs", "radar"]
    if not isinstance(keys, list):
        keys = list(keys)

    times = {}
    time_min = None
    time_max = None
    for key in keys:
        files = list(Path(path).glob(f"**/{key}*.nc"))
        times_k = [
            datetime.strptime(name.name[len(key) + 1:-3], "%Y%m%d_%H_%M") for name in files
            ]

        times_k = xr.DataArray(times_k)
        t_min = times_k.min()
        time_min = min(time_min, t_min) if time_min is not None else t_min
        t_max = times_k.max()
        time_max = max(time_max, t_max) if time_max is not None else t_max

        times[key] = times_k

    figure = plt.figure(figsize=(6, 4))
    ax = figure.add_subplot(1, 1, 1)

    bins = np.arange(
        time_min.astype("datetime64[D]").data,
        (time_max + np.timedelta64(2, "D")).astype("datetime64[D]").data,
        dtype="datetime64[D]"
    )
    x = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])

    for key in keys:
        y, _ = np.histogram(times[key], bins=bins)
        ax.plot(x, y, label=key)

    ax.legend()



class TestDataSet:
    def __init__(self,
                 sequence_length=1,
                 size=(128, 128),
                 input_sampling=1
    ):
        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.sequence_length = sequence_length
        self.input_sampling = input_sampling

    def __len__(self):
        return 10_000

    def init_function(self, w_id):
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, index):

        xs = [self.rng.uniform(-1, 1) * np.ones((11,) + self.size, dtype=np.float32)
              for i in range(self.sequence_length)]
        y = np.stack(xs, axis=0)[:, 0]
        y[1:] += y[:-1]
        y[1:] *= 0.5

        y = list(y)
        xs = [{"geo": x[..., ::self.input_sampling, ::self.input_sampling]} for x in xs]
        for x in xs:
            x = x["geo"]
            x += self.rng.uniform(-0.01, 0.01, size=x.shape)

        return xs, y
