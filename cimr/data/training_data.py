"""
cimr.data.training_data
=======================

The CIMR training data.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from quantnn.normalizer import MinMaxNormalizer
import torch
from torch import nn
import xarray as xr

from cimr.areas import NORDIC_2
from cimr.data.baltrad import Baltrad
from cimr.data.seviri import SEVIRI
from cimr.data.mhs import MHS
from cimr.data.avhrr import AVHRR
from cimr.utils import MISSING

NORMALIZER_SEVIRI = MinMaxNormalizer(np.ones((1, 12, 1, 1)), feature_axis=0)
NORMALIZER_SEVIRI.stats = {
    0: (0.0, 93.5),
    1: (0.0, 78.4),
    2: (0.0, 89.200005),
    3: (0.0, 82.200005),
    4: (-3.0, 307.5),
    5: (0.0, 239.90001),
    6: (0.0, 252.8),
    7: (0.0, 286.7),
    8: (0.0, 248.7),
    9: (0.0, 291.2),
    10: (0.0, 288.9),
    11: (0.0, 258.0),
}


NORMALIZER_AVHRR = MinMaxNormalizer(np.ones((1, 5, 1, 1)), feature_axis=0)
NORMALIZER_AVHRR.stats = {
    0: (0.0, 252.54),
    1: (-323.93, 317.27),
    2: (-297.4, 339.07),
    3: (210.7, 336.44),
    4: (216.19, 295.59),
}


NORMALIZER_MHS = MinMaxNormalizer(np.ones((1, 5, 1, 1)), feature_axis=0)
NORMALIZER_MHS.stats = {
    0: (0.012644, 0.0204179),
    1: (-0.0428406, 0.16728291),
    2: (0.0677726, 0.0796296),
    3: (0.069345, 0.0804358),
    4: (0.0675721, 0.089615),
}


@dataclass
class SampleRecord:
    baltrad: Path = None
    seviri: Path = None
    mhs: Path = None
    avhrr: Path = None


def get_date(filename):
    _, yearmonthday, hour, minute = filename.stem.split("_")
    year = yearmonthday[:4]
    month = yearmonthday[4:6]
    day = yearmonthday[6:]
    return np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}:00")


class CIMRDataset:
    def __init__(self, folder, sample_rate=4):
        self.folder = Path(folder)
        self.sample_rate = sample_rate

        baltrad_files = sorted(list((self.folder / "radar").glob("radar*.nc")))
        times = np.array(list(map(get_date, baltrad_files)))

        self.samples = {
            time: SampleRecord(b_file) for time, b_file in zip(times, baltrad_files)
        }

        seviri_files = sorted(list((self.folder / "seviri").glob("seviri*.nc")))
        times = np.array(list(map(get_date, seviri_files)))
        for time, seviri_file in zip(times, seviri_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.seviri = seviri_file

        mhs_files = sorted(list((self.folder / "mhs").glob("mhs*.nc")))
        times = np.array(list(map(get_date, mhs_files)))
        for time, mhs_file in zip(times, mhs_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.mhs = mhs_file

        avhrr_files = sorted(list((self.folder / "avhrr").glob("avhrr*.nc")))
        times = np.array(list(map(get_date, avhrr_files)))
        for time, avhrr_file in zip(times, avhrr_files):
            sample = self.samples.get(time)
            if sample is not None:
                sample.avhrr = avhrr_file

        self.keys = np.array(list(self.samples.keys()))

        self.window_size = 128

    def __len__(self):
        return len(self.samples) * self.sample_rate

    def __getitem__(self, index):

        scene_index = index // self.sample_rate
        key = self.keys[scene_index]

        with xr.open_dataset(self.samples[key].baltrad) as data:

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

                if (y_s >= 0).mean() > 0.2:
                    found = True

        y_s = np.nan_to_num(y_s, nan=-100)
        y = torch.tensor(y_s, dtype=torch.float)

        x = {}

        # SEVIRI data
        if self.samples[key].seviri is not None:
            with xr.open_dataset(self.samples[key].seviri) as data:
                row_slice = slice(i_start * 2, i_end * 2)
                col_slice = slice(j_start * 2, j_end * 2)
                xs = np.stack(
                    [
                        data[f"channel_{i:02}"].data[row_slice, col_slice]
                        for i in range(1, 13)
                    ]
                )
            x["seviri"] = torch.tensor(NORMALIZER_SEVIRI(xs), dtype=torch.float)
        else:
            x["seviri"] = torch.tensor(
                MISSING * np.ones((12,) + (self.window_size // 2,) * 2),
                dtype=torch.float,
            )

        # AVHRR data
        if self.samples[key].avhrr is not None:
            with xr.open_dataset(self.samples[key].avhrr) as data:
                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)
                xs = np.stack(
                    [
                        data[f"channel_{i:01}"].data[row_slice, col_slice]
                        for i in range(1, 6)
                    ]
                )
            x["avhrr"] = torch.tensor(NORMALIZER_AVHRR(xs), dtype=torch.float)
        else:
            x["avhrr"] = torch.tensor(
                MISSING * np.ones((5,) + (self.window_size,) * 2), dtype=torch.float
            )

        # MHS data
        if self.samples[key].mhs is not None:
            with xr.open_dataset(self.samples[key].mhs) as data:
                row_slice = slice(i_start, i_end)
                col_slice = slice(j_start, j_end)
                xs = np.stack(
                    [
                        data[f"channel_{i:02}"].data[row_slice, col_slice]
                        for i in range(1, 6)
                    ]
                )
            x["mhs"] = torch.tensor(NORMALIZER_MHS(xs), dtype=torch.float)
        else:
            x["mhs"] = torch.tensor(
                MISSING * np.ones((5,) + (self.window_size // 4,) * 2),
                dtype=torch.float,
            )

        return x, y

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
        ax.set_title("Baltrad")

        ax = axs[0, 1]
        ax.set_title("Seviri 12u")

        ax = axs[1, 0]
        ax.set_title("AVHRR")

        ax = axs[1, 1]
        ax.set_title("MHS (89 GHz)")

        def animator(frame):

            sample = self.samples[keys[frame]]

            baltrad_data = xr.load_dataset(sample.baltrad)
            ax = axs[0, 0]
            dbz = baltrad_data.dbz.data.copy()
            dbz[baltrad_data.qi < 0.8] = np.nan
            img = ax.imshow(dbz, extent=extent, vmin=-20, vmax=20)
            ax.coastlines(color="grey")

            ax = axs[0, 1]
            if sample.seviri is not None:
                seviri_data = xr.load_dataset(sample.seviri)
                ax.imshow(seviri_data.channel_10, extent=extent)
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
        with xr.open_dataset(self.samples[key].baltrad) as data:
            y = data.dbz.data.copy()
            y[data.qi < 0.8] = np.nan

        y = torch.tensor(y, dtype=torch.float)
        shape = y.shape

        x = {}
        # SEVIRI data
        if self.samples[key].seviri is not None:
            with xr.open_dataset(self.samples[key].seviri) as data:
                xs = np.stack([data[f"channel_{i:02}"].data for i in range(1, 13)])
            x["seviri"] = torch.tensor(NORMALIZER_SEVIRI(xs), dtype=torch.float)
        else:
            x["seviri"] = torch.tensor(
                MISSING * np.ones((12,) + tuple([s // 2 for s in shape])),
                dtype=torch.float,
            )

        # AVHRR data
        if self.samples[key].avhrr is not None:
            with xr.open_dataset(self.samples[key].avhrr) as data:
                xs = np.stack([data[f"channel_{i:01}"].data for i in range(1, 6)])
            x["avhrr"] = torch.tensor(NORMALIZER_AVHRR(xs), dtype=torch.float)
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
        input_seviri = x["seviri"]
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
        print(padding)

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
        print(padding)

        input_seviri = nn.functional.pad(input_seviri, padding, "replicate")
        x["seviri"] = input_seviri

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
        print(padding)

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

            x["seviri"] = x["seviri"].unsqueeze(0)
            x["avhrr"] = x["avhrr"].unsqueeze(0)
            x["mhs"] = x["mhs"].unsqueeze(0)

            print(x["seviri"].shape)
            print(x["avhrr"].shape)
            print(x["mhs"].shape)

            yield x, y, slice_y, slice_x
