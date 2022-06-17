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
import xarray as xr

from cimr.areas import NORDIC_2
from cimr.data.baltrad import Baltrad
from cimr.data.seviri import SEVIRI
from cimr.data.mhs import MHS
from cimr.data.avhrr import AVHRR

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
    def __init__(
            self,
            folder,
            sample_rate=4):
        self.folder = Path(folder)
        self.sample_rate = sample_rate

        baltrad_files = sorted(list((self.folder / "radar").glob("radar*.nc")))
        times = np.array(list(map(get_date, baltrad_files)))

        self.samples = {
            time: SampleRecord(b_file)
            for time, b_file in zip(times, baltrad_files)
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

            y = data.dbz.copy()
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

                y = y.data[row_slice, col_slice]

                if (y > 0).mean() > 0.2:
                    found = True

        x = {}

        # SEVIRI data
        if self.samples[key].seviri is not None:
            with xr.open_dataset(self.samples[key].seviri) as data:
                row_slice = slice(i_start * 2, i_end * 2)
                col_slice = slice(j_start * 2, j_end * 2)
                xs = np.stack([
                    data[f"channel_{i:02}"].data[row_slice, col_slice]
                    for i in range(1, 13)
                ])
            x["seviri"] = xs

        # AVHRR data
        if self.samples[key].avhrr is not None:
            with xr.open_dataset(self.samples[key].avhrr) as data:
                row_slice = slice(4 * i_start, 4 * i_end)
                col_slice = slice(4 * j_start, 4 * j_end)
                xs = np.stack([
                    data[f"channel_{i:01}"].data[row_slice, col_slice]
                    for i in range(1, 6)
                ])
            x["avhrr"] = xs

        # MHS data
        if self.samples[key].mhs is not None:
            with xr.open_dataset(self.samples[key].mhs) as data:
                row_slice = slice(i_start, i_end)
                col_slice = slice(j_start, j_end)
                xs = np.stack([
                    data[f"channel_{i:02}"].data[row_slice, col_slice]
                    for i in range(1, 6)
                ])
            x["mhs"] = xs

        return x, y

    def make_animator(self, start_time, end_time):
        indices =  (self.keys >= start_time) * (self.keys <= end_time)
        keys = self.keys[indices]

        crs = NORDIC_2.to_cartopy_crs()
        extent = NORDIC_2.area_extent
        extent = (extent[0], extent[2], extent[1], extent[3])

        f = plt.figure(figsize=(10, 12))
        gs = GridSpec(2, 2)
        axs = np.array([
            [f.add_subplot(gs[i, j], projection=crs) for j in range(2)]
            for i in range(2)
        ])

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
            img = ax.imshow(dbz, extent=extent)
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










