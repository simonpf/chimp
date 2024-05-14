"""
chimp.data.training_data
=======================

This module defines the CHIMP training data classes for loading single time-step
input data and sequence data.
"""
from dataclasses import dataclass
from datetime import datetime
import logging
from math import floor, ceil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import numpy as np
from rich.progress import Progress
from scipy import signal
from scipy import fft
from scipy import ndimage
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision
from torchvision.transforms.functional import center_crop
import xarray as xr
import pandas as pd

from pytorch_retrieve.tensors.masked_tensor import MaskedTensor


from chimp import data
from chimp.definitions import MASK
from chimp.utils import get_date
from chimp.data import (
    get_input_dataset,
    get_reference_dataset,
    InputDataset,
    ReferenceDataset
)
from chimp.data import input, reference


LOGGER = logging.getLogger(__name__)


class SingleStepDataset(Dataset):
    """
    PyTorch Dataset to load CHIMP training data for single-time-step retrievals.
    """
    def __init__(
        self,
        path: Path,
        input_datasets: List[Union[str, InputDataset]],
        reference_datasets: List[Union[str, ReferenceDataset]],
        sample_rate: float = 1.0,
        scene_size: int = 128,
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        augment: bool = True,
        time_step: Optional[np.timedelta64] = None,
        validation: bool = False,
        quality_threshold: Union[float, List[float]] = 0.8,
        require_ref_data: bool = True
    ):
        """
        Args:
            path: The root directory containing the training data.
            input_datasets: List of the input datasets or their names from which
                to load the retrieval input data.
            reference_datasets: List of the reference datasets or their names
                from which to load the reference data.
            sample_rate: How often each scene should be sampled per epoch.
            scene_size: Size of the training scenes. If 'scene_size' < 0, the full
            input data will be loaded.
            start_time: Start time of time interval to which to restrict training data.
            end_time: End time of a time interval to which to restrict the training data.
            augment: Whether to apply random transformations to the training inputs.
            time_step: Minimum time step between consecutive reference samples.
                Can be used to sub-sample the reference data.
            validation: If 'True', repeated sampling will reproduce identical scenes.
            quality_threshold: Thresholds for the quality indices applied to limit
                reference data pixels.
            require_ref_data: If 'True' only samples with corresponding reference
                data are considered.
        """
        self.path = Path(path)

        self.input_datasets = np.array([
            get_input_dataset(input_dataset) for input_dataset in input_datasets
        ])
        self.reference_datasets = np.array([
            get_reference_dataset(reference_dataset)
            for reference_dataset in reference_datasets
        ])

        self.sample_rate = sample_rate
        self.augment = augment

        n_input_datasets = len(self.input_datasets)
        n_reference_datasets = len(self.reference_datasets)
        n_datasets = n_input_datasets + n_reference_datasets

        sample_files = {}
        for ref_ind, reference_dataset in enumerate(self.reference_datasets):
            files = reference_dataset.find_training_files(self.path)
            times = np.array(list(map(get_date, files)))
            for time, filename in zip(times, files):
                files = sample_files.setdefault(time, ([None] * n_datasets))
                files[ref_ind] = filename

        if len(sample_files) == 0:
            raise RuntimeError(
                f"Found no reference data files in path '{self.path}' for "
                f" reference datasets '{[ds.name for ds in self.reference_datasets]}'."
            )

        for input_ind, input_dataset in enumerate(self.input_datasets):
            input_files = input_dataset.find_training_files(self.path)
            times = np.array(list(map(get_date, input_files)))
            for time, input_file in zip(times, input_files):
                if time in sample_files or not require_ref_data:
                    files = sample_files.setdefault(time, [None] * n_datasets)
                    files[n_reference_datasets + input_ind] = input_file

        self.base_scale = min(
            [reference_dataset.scale for reference_dataset in self.reference_datasets]
        )
        self.max_scale = 0
        self.scales = {}
        for input_dataset in self.input_datasets:
            scale = input_dataset.scale / self.base_scale
            self.scales[input_dataset.name] = scale
            self.max_scale = max(self.max_scale, scale)

        times = np.array(list(sample_files.keys()))
        samples = np.array(list(sample_files.values()))
        reference_files = samples[:, :n_reference_datasets]

        inds = np.argsort(times)
        times = times[inds]
        samples = samples[inds]
        reference_files = reference_files[inds]

        input_files = samples[:, n_reference_datasets:]

        if start_time is not None and end_time is not None:
            indices = (times >= start_time) * (times < end_time)
            times = times[indices]
            reference_files = reference_files[indices]
            input_files = input_files[indices]

        self.times = times
        self.reference_files = reference_files
        self.input_files = input_files

        if time_step is None:
            time_step = np.min(np.diff(times))
        self.time_step = time_step

        # Ensure that data is consistent
        assert len(self.times) == len(self.reference_files)
        assert len(self.times) == len(self.input_files)

        self.full = False
        if scene_size < 0:
            ref_dataset = np.argmin(
                [reference_dataset.scale for reference_dataset in self.reference_datasets]
            )
            ref_file = np.where(reference_files[:, ref_dataset])[0][0]
            ref_file = reference_files[ref_file, ref_dataset]
            with xr.open_dataset(ref_file) as ref_data:
                scene_size = tuple(ref_data.sizes.values())[:2]
            self.full = True

        self.scene_size = scene_size
        self.validation = validation
        if self.validation:
            self.augment = False
        self.init_rng()

        if isinstance(quality_threshold, float):
            quality_threshold = [quality_threshold] * n_reference_datasets
        self.quality_threshold = np.array(quality_threshold)


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

    def worker_init_fn(self, *args):
        """
        Pytorch retrieve interface.
        """
        return self.init_rng(*args)

    def load_reference_sample(
        self,
        files: Tuple[Path],
        slices: Optional[Tuple[int]],
        scene_size: int,
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
            scene_size: The window size specified with respect to the
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
        if isinstance(scene_size, int):
            scene_size = (scene_size,) * 2

        if slices is not None:
            i_start, i_end, j_start, j_end = slices
            row_slice = slice(i_start, i_end)
            col_slice = slice(j_start, j_end)
        else:
            row_slice = slice(0, None)
            col_slice = slice(0, None)

        # Load reference data.
        y = {}
        for dataset_ind, reference_dataset in enumerate(self.reference_datasets):
            y.update(
                reference_dataset.load_sample(
                    files[dataset_ind],
                    scene_size,
                    self.base_scale,
                    slices,
                    self.rng,
                    rotate=rotate,
                    flip=flip,
                    quality_threshold=self.quality_threshold[dataset_ind]
                )
            )
        return y


    def load_input_sample(
        self,
        files: Tuple[Path],
        slices: Optional[Tuple[int]],
        scene_size: int,
        forecast: bool = False,
        rotate: Optional[float] = None,
        flip: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Load input for given training sample.

        Args:
            files: Numpy array containing the paths to the files to load.
            slices: Tuple ``(i_start, i_end, j_start, j_end)`` defining
                defining the crop of the domain. If set to 'None', the full
                domain is loaded.
            scene_size: The window size specified with respect to the
                reference data.
            forecast: If 'True', no input data will be loaded and all inputs
                will be set to None.
            rotate: If provided, should be float specifying the degree by which
                to rotate the input.
            flip: If 'True', input will be flipped along the last axis.

        Return:
            A dictionary mapping input dataset names to corresponding input
            data tensors.
        """
        if isinstance(scene_size, int):
            scene_size = (scene_size,) * 2

        if slices is not None:
            i_start, i_end, j_start, j_end = slices
            row_slice = slice(i_start, i_end)
            col_slice = slice(j_start, j_end)
        else:
            row_slice = slice(0, None)
            col_slice = slice(0, None)

        x = {}
        for input_ind, input_dataset in enumerate(self.input_datasets):
            input_file = files[input_ind]
            x_s = input_dataset.load_sample(
                input_file,
                scene_size,
                self.base_scale,
                slices,
                self.rng,
                rotate=rotate,
                flip=flip,
            )
            x[input_dataset.input_name] = x_s
        return x

    def __len__(self):
        """Number of samples in dataset."""
        return floor(len(self.times) * self.sample_rate)

    def __getitem__(self, index):
        """Return ith training sample."""
        n_samples = len(self.times)
        sample_index = min(floor(index / self.sample_rate), n_samples)
        if self.augment:
            limit = min(floor((index + 1) / self.sample_rate), n_samples)
            if limit > sample_index:
                sample_index = self.rng.integers(sample_index, limit)


        # We load a larger window when input is rotated to avoid
        # missing values.
        if self.augment:
            if not self.full:
                scene_size = int(1.42 * self.scene_size)
                rem = scene_size % self.max_scale
                if rem != 0:
                    scene_size += self.max_scale - rem
                ang = -180 + 360 * self.rng.random()
            else:
                scene_size = self.scene_size
                ang = None
            flip = self.rng.random() > 0.5
        else:
            scene_size = self.scene_size
            ang = None
            flip = False

        try:
            if not self.full:
                rd_ind = np.where(self.reference_files[sample_index])[0][0]
                slices = self.reference_datasets[rd_ind].find_random_scene(
                    self.reference_files[sample_index][rd_ind],
                    self.rng,
                    multiple=4,
                    scene_size=scene_size,
                    quality_threshold=self.quality_threshold[rd_ind]
                )
                if slices is None:
                    LOGGER.warning(
                        " Couldn't find a scene in reference file '%s' satisfying "
                        "the quality requirements. Falling back to another "
                        "radomly-chosen reference data file.",
                        self.reference_files[sample_index][0]
                    )
                    new_ind = self.rng.integers(0, len(self))
                    return self[new_ind]
            else:
                slices = (0, scene_size[0], 0, scene_size[1])

            x = self.load_input_sample(
                self.input_files[sample_index], slices, self.scene_size, rotate=ang, flip=flip
            )
            y = self.load_reference_sample(
                self.reference_files[sample_index], slices, self.scene_size, rotate=ang, flip=flip
            )
        except Exception:
            LOGGER.exception(
                f"Loading of training sample for '%s'"
                "failed. Falling back to another radomly-chosen step.",
                self.times[sample_index]
            )
            new_ind = self.rng.integers(0, len(self))
            return self[new_ind]

        return x, y

    def _plot_sample_frequency(
            self,
            datasets,
            files,
            ax=None,
            temporal_resolution="M",
    ):
        """
        Plot sample frequency of datasets.

        Args:
            datasets: A list of dataset types defining the input or reference datasets in the training
                data.
            files:
        """
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(6, 4))


        times = self.times
        start_time = times[0]
        end_time = times[-1]

        if isinstance(temporal_resolution, str):
            time_step = np.timedelta64(1, temporal_resolution).astype('timedelta64[s]')
        else:
            time_step = temporal_resolution.astype("timedelta64[s]")
        bins = np.arange(
            start_time,
            end_time + 2 * time_step,
            time_step,
        )

        acc = None
        x = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])

        n_ds = len(datasets)
        norm = Normalize(0, n_ds)
        cmap = ScalarMappable(norm=norm, cmap="plasma")

        for ind, dataset in enumerate(datasets):
            weights = (files[:, ind] != None).astype(np.float32)
            cts = np.histogram(times, weights=weights, bins=bins)[0]

            color = cmap.to_rgba(ind)

            if acc is None:
                ax.fill_between(x, cts, label=dataset.name, facecolor=color, edgecolor="none", alpha=0.7, linewidth=2)
                acc = cts
            else:
                ax.fill_between(x, acc, acc + cts, label=dataset.name, facecolor=color, edgecolor="none", alpha=0.7, linewidth=2)
                acc += cts

        return ax

    def plot_input_sample_frequency(
            self,
            ax=None,
            temporal_resolution="M"
    ):
        self._plot_sample_frequency(self.input_datasets, self.input_files, ax=ax, temporal_resolution=temporal_resolution)

    def plot_reference_sample_frequency(
            self,
            ax=None,
            temporal_resolution="M"
    ):
        self._plot_sample_frequency(self.reference_datasets, self.reference_files, ax=ax, temporal_resolution=temporal_resolution)


    def plot_reference_data_availability(
            self,
            reference_dataset: str,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Plot available reference data pixels by channel for a given input.

        Args:
            reference_dataset: The name of the reference dataset for which to
                 plot the sample availability.

        Return:
            A tuple containing the matplotlib.Figure and matplotlib.Axes objects
            containing the curves representing the number of valid reference data
            samples per time step.
        """
        ref_names = [ref.name for ref in self.reference_datasets]
        ind = ref_names.index(reference_dataset)
        ref = self.reference_datasets[ind]
        time_step = np.min(np.diff(self.times))

        start_time = self.times.min()
        end_time = self.times.max()
        time_steps = np.arange(start_time, end_time, 1.01 * time_step)

        times = np.array([
            time for path, time in zip(self.reference_files[:, ind], self.times)
            if path is not None
        ])
        files = [
            path for path in self.reference_files[:, ind] if path is not None
        ]


        counts = {}
        scene_size = None

        with Progress() as progress:

            task = progress.add_task(
                "Calculating valid samples:", total=len(files)
            )

            for ind, (time, path) in enumerate(zip(times, files)):

                t_ind = np.digitize(time.astype("int64"), time_steps.astype("int64"))

                try:
                    if scene_size is None:
                        with xr.open_dataset(path) as scene:
                            scene_size = tuple(scene.sizes.values())[:2]

                    data = ref.load_sample(
                        path,
                        scene_size,
                        ref.scale,
                        (slice(0, scene_size[0]), slice(0, scene_size[1])),
                        None
                    )
                    for key, data_t in data.items():
                        cts = counts.setdefault(key, np.zeros(len(time_steps)))
                        cts[t_ind] = torch.isfinite(data_t).sum()

                except Exception:
                    LOGGER.exception(
                        "Encountered an error opening file %s.",
                        path
                    )

                progress.update(task, advance=1)

        fig = plt.Figure(figsize=(20, 4))
        gs = GridSpec(1, 1)

        ax = fig.add_subplot(gs[0, 0])

        norm = Normalize(0, len(cts))
        cmap = ScalarMappable(norm=norm, cmap="plasma")

        for ind, (key, cts) in enumerate(counts.items()):
            clr = cmap.to_rgba(ind)
            ax.plot(time_steps, cts, label=key, c=clr)

        ax.set_ylabel("# valid pixels")
        for label in ax.get_xticklabels():
            label.set_rotation(90)

        ax.set_xlim(start_time, end_time)

        return fig, ax


    def plot_input_data_availability(
            self,
            input_name: str,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Plot available input pixels by channel for a given input.

        Args:
            input_name: The name of the input dataset.

        Return:
            A tuple containing the matplotlib.Figure and matplotlib.Axes objects
            containing the curves representing the number of valid inputs per
            time step.
        """
        input_names = [inpt.name for inpt in self.input_datasets]
        ind = input_names.index(input_name)
        inpt = self.input_datasets[ind]
        time_step = np.min(np.diff(self.times))

        start_time = self.times.min()
        end_time = self.times.max()
        time_steps = np.arange(start_time, end_time, 1.01 * time_step)

        times = np.array([
            time for path, time in zip(self.input_files[:, ind], self.times)
            if path is not None
        ])
        files = [
            path for path in self.input_files[:, ind] if path is not None
        ]


        n_chans = inpt.n_channels
        counts = np.zeros((n_chans, len(time_steps)))

        scene_size = None

        with Progress() as progress:

            task = progress.add_task(
                "Calculating valid samples:", total=len(files)
            )

            for ind, (time, path) in enumerate(zip(times, files)):

                t_ind = np.digitize(time.astype("int64"), time_steps.astype("int64"))

                try:
                    if scene_size is None:
                        with xr.open_dataset(path) as scene:
                            scene_size = tuple(scene.sizes.values())[:2]

                    data = inpt.load_sample(
                        path,
                        scene_size,
                        inpt.scale,
                        (slice(0, scene_size[0]), slice(0, scene_size[1])),
                        None
                    )
                    for ch_ind in range(n_chans):
                        counts[ch_ind, t_ind] = np.isfinite(data[ch_ind]).sum()
                except Exception:
                    LOGGER.error(
                        "Encountered an error opening file %s.",
                        path
                    )

                progress.update(task, advance=1)

        fig = plt.Figure(figsize=(20, 4))
        gs = GridSpec(1, 2, width_ratios=[1.0, 0.03])

        ax = fig.add_subplot(gs[0, 0])

        counts = np.cumsum(counts, 0)
        norm = Normalize(0, n_chans)
        cmap = ScalarMappable(norm=norm, cmap="plasma")

        lower = 0.0
        for ch_ind in range(n_chans):
            color = cmap.to_rgba(ch_ind)
            ax.fill_between(
                time_steps,
                lower,
                counts[ch_ind],
                facecolor=color
            )
            lower = counts[ch_ind]

        ax.set_xlim(time_steps[0], time_steps[-1])
        ax.set_ylim(0, counts[ch_ind].max())

        ax.set_ylabel("# valid pixels")
        for label in ax.get_xticklabels():
            label.set_rotation(90)

        ax = fig.add_subplot(gs[0, 1])
        plt.colorbar(cmap, cax=ax, label="Channel #")
        ax.set_xlim(start_time, end_time)

        return fig, ax



class SingleStepPretrainDataset(SingleStepDataset):
    """
    PyTorch dataset class for single-time-step retrievals with training
    samples resampled to ensure uniform sampling of input observations.
    """
    def __init__(
        self,
        path: Path,
        input_datasets: List[Union[str, InputDataset]],
        reference_datasets: List[Union[str, ReferenceDataset]],
        sample_rate: float = 1.0,
        scene_size: int = 128,
        start_time: Optional[np.datetime64] = None,
        end_time: Optional[np.datetime64] = None,
        augment: bool = True,
        time_step: Optional[np.timedelta64] = None,
        validation: bool = False,
        quality_threshold: Union[float, List[float]] = 0.8
    ):
        super().__init__(
            path=path,
            input_datasets=input_datasets,
            reference_datasets=reference_datasets,
            sample_rate=sample_rate,
            scene_size=scene_size,
            start_time=start_time,
            end_time=end_time,
            augment=augment,
            time_step=time_step,
            validation=validation,
            quality_threshold=quality_threshold
        )


        samples_by_input = [[] for _ in self.input_datasets]
        for scene_index in range(len(self.times)):
            input_files = self.input_files[scene_index]
            for input_ind in range(len(self.input_datasets)):
                # Input not available at time step
                if input_files[input_ind] is None:
                    continue
                samples_by_input[input_ind].append(scene_index)

        tot_samples = super().__len__()
        self.samples_per_input = ceil(tot_samples / len(self.input_datasets))
        self.samples_by_input = [
            np.random.permutation(smpls) for smpls in samples_by_input
        ]

    def __len__(self):
        return int(
            self.samples_per_input * len(self.input_datasets) * self.sample_rate
        )

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        n_samples = self.samples_per_input * len(self.input_datasets)
        input_index = ind // self.samples_per_input
        input_dataset = self.input_datasets[input_index]
        input_samples = self.samples_by_input[input_index]
        scene_index = ind % self.samples_per_input
        if self.augment:
            limit = floor(scene_index * len(input_samples) / self.samples_per_input)
            if limit > scene_index:
                scene_index = self.rng.integers(scene_index, limit)

        scene_index = input_samples[scene_index % len(input_samples)]


        # We load a larger window when input is rotated to avoid
        # missing values.
        if self.augment:
            if not self.full:
                scene_size = int(1.42 * self.scene_size)
                rem = scene_size % self.max_scale
                if rem != 0:
                    scene_size += self.max_scale - rem
                ang = -180 + 360 * self.rng.random()
            else:
                scene_size = self.scene_size
                ang = None
            flip = self.rng.random() > 0.5
        else:
            scene_size = self.scene_size
            ang = None
            flip = False

        try:
            if not self.full:
                rel_scale = input_dataset.scale / self.base_scale
                slices = input_dataset.find_random_scene(
                    self.input_files[scene_index][input_index],
                    self.rng,
                    multiple=self.max_scale / input_dataset.scale,
                    scene_size=scene_size / rel_scale,
                )
                if slices is None:
                    LOGGER.warning(
                        " Couldn't find a scene in input file '%s' containing "
                        "valid input observations. Falling back to another "
                        "radomly-chosen scene.",
                        self.input_files[scene_index][input_index]
                    )
                    start = input_index * self.samples_per_input
                    end = start + self.samples_per_input
                    new_ind = self.rng.integers(start, end)
                    return self[new_ind]
            else:
                slices = (0, scene_size[0], 0, scene_size[1])

            slices = data.utils.scale_slices(slices, self.base_scale / input_dataset.scale)
            slices = (slices[0].start, slices[0].stop, slices[1].start, slices[1].stop)

            x = self.load_input_sample(
                self.input_files[scene_index], slices, self.scene_size, rotate=ang, flip=flip
            )
            y = self.load_reference_sample(
                self.reference_files[scene_index], slices, self.scene_size, rotate=ang, flip=flip
            )

            no_ref_data = all([tensor.isnan().all() for tensor in y.values()])
            if no_ref_data:
                LOGGER.warning(
                    "No reference data in file '%s' ",
                    self.input_files[scene_index][input_index]
                )
                start = input_index * self.samples_per_input
                end = start + self.samples_per_input
                new_ind = self.rng.integers(start, end)
                return self[new_ind]

        except Exception:
            LOGGER.exception(
                f"Loading of training sample for '%s'"
                "failed. Falling back to another radomly-chosen step.",
                self.times[scene_index]
            )
            start = input_index * self.samples_per_input
            end = start + self.samples_per_input
            new_ind = self.rng.integers(start, end)
            return self[new_ind]

        return x, y


def expand_times_and_files(
        times: np.ndarray,
        input_files: np.ndarray,
        reference_files: np.ndarray,
        time_step: Optional[np.timedelta64] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand temporally sparse records of time stamps and corresponding input
    and reference files to time-uniform intervals.

    Args:
        times: An array containing the sorted time stamps of available samples.
        input_files: An array containing the input files corresponding to
            the time steps in 'times'.
        reference_files: An array containing the reference files corresponding
            to the time steps in 'times'.
        time_step: The temporal resolution of time steps in 'times'. Will be
            determined as the minimum time difference between consecutive samples
            in 'times'.

    Return:
        A tuple ``(times_full, input_files_full, reference_files_full)`` containing
        the input arrays 'times', 'input_files', and 'reference_files' expanded
        to temporally dense sampling.
    """
    if time_step is None:
        time_step = np.diff(times).min()

    start_time = times.min()
    end_time = times.max()
    times_full = np.arange(start_time, end_time + 0.5  * time_step, time_step)
    n_steps = times_full.size

    inds = (times - start_time) // time_step
    input_files_full = np.zeros(
        (n_steps,) + input_files.shape[1:],
        dtype="object"
    )
    input_files_full[:] = None
    input_files_full[inds] = input_files

    reference_files_full = np.zeros(
        (n_steps,) + reference_files.shape[1:],
        dtype="object"
    )
    reference_files_full[:] = None
    reference_files_full[inds] = reference_files

    return times_full, input_files_full, reference_files_full


def find_sequence_starts_and_ends(
        input_files: np.ndarray,
        reference_files: np.ndarray,
        sequence_length: int,
        forecast: int,
        include_input_steps: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine consecutive indices of the first and last input samples that have
    valid reference data at the end of the sequence window and include all
    reference samples at most once.

    Args:
        input_files: Array containing the input files for each time step.
        reference_files: Array containing the reference files for each time
            step.
        sequence_length: The length of input sequences.
        forecast: The number of forecast steps.
        include_input_steps: Whether or not the input steps are included in the
            reference data.
    """

    valid_inputs = np.any(input_files != None, -1).astype(np.float32)
    k = np.ones(2 * sequence_length - 1)
    k[sequence_length:] = 0.0
    thresh = k.sum() * 0.5
    valid_inputs = signal.convolve(valid_inputs, k, mode="same", method="direct") >= thresh
    valid_inputs= valid_inputs[:-(sequence_length + forecast - 1)]

    valid_reference = np.any(reference_files != None, -1).astype(np.float32)
    k = np.ones(2 * (sequence_length + forecast) - 1)
    if include_input_steps:
        k[sequence_length + forecast:] = 0.0
    else:
        k[forecast:] = 0.0
    valid_reference = signal.convolve(
        valid_reference,
        k,
        mode="same",
        method="direct"
    ) > 0.0
    valid_reference = valid_reference[:-(sequence_length + forecast - 1)]

    valid_inds = np.where(valid_inputs * valid_reference)[0]
    starts = []
    ends = []

    while len(valid_inds) > 0:
        curr_start = valid_inds[0]
        starts.append(curr_start)

        if include_input_steps:
            tot_len = sequence_length
        else:
            tot_len = forecast

        vref = valid_reference[curr_start: curr_start + tot_len]
        wlen = np.where(vref)[0].max()
        ends.append(curr_start + wlen)
        valid_inds = valid_inds[valid_inds > curr_start + wlen]

    return np.array(starts), np.array(ends)


class SequenceDataset(SingleStepDataset):
    """
    Dataset class for temporal merging of satellite observations.
    """

    def __init__(
        self,
        path: Path,
        input_datasets: List[str],
        reference_datasets: List[str],
        sample_rate: int = 2,
        scene_size: int = 256,
        sequence_length: int = 32,
        forecast: int = 0,
        forecast_range: Optional[int] = None,
        include_input_steps: bool = True,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
        augment: bool = True,
        shrink_output: Optional[int] = None,
        validation: bool = False,
        time_step: Optional[np.timedelta64] = None,
        quality_threshold: float = 0.8,
    ):
        """
        Args:
            path: The path to the training data.
            input_datasets: List of input datasets or their names from which to load
                 the input data.
            reference_datasets: List of reference datasets or their names from which
                 to load the reference data.
            sample_rate: Rate for oversampling of training scenes.
            scene_size: The size of the input data.
            sequence_length: The length of input data sequences.
            forecast: The number of time steps to forecast.
            include_input_steps: Whether reference data for the input steps should
                be loaded as well.
            start_time: Optional start time to limit the samples.
            end_time: Optional end time to limit the available samples.
            augment: Whether to apply random transformations to the training
                inputs.
            shrink_output: If given, the reference data scenes will contain
                only the center crop the total scene with the size of the
                crop calculated by dividing the input size by the given factor.
            validation: If 'True' sampling will reproduce identical scenes.
            time_step: Optional time step to sub-sample the input data.
            quality_threshold: Thresholds for the quality indices applied to limit
                reference data pixels.
        """
        super().__init__(
            path,
            input_datasets,
            reference_datasets,
            sample_rate=sample_rate,
            scene_size=scene_size,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
            quality_threshold=quality_threshold,
            augment=augment,
            validation=validation,
            require_ref_data=False
        )

        self.sequence_length = sequence_length
        self.forecast = forecast
        if forecast_range is None:
            forecast_range = forecast
        self.forecast_range = forecast_range
        total_length =  sequence_length + forecast_range
        self.total_length = total_length
        self.include_input_steps = include_input_steps
        self.shrink_output = shrink_output

        full = expand_times_and_files(
            self.times,
            self.input_files,
            self.reference_files,
            time_step=time_step
        )
        self.times, self.input_files, self.reference_files = full
        self.valid_ref = np.any(self.reference_files != None, -1)
        seqs = find_sequence_starts_and_ends(
            self.input_files,
            self.reference_files,
            self.sequence_length,
            self.forecast_range,
            self.include_input_steps
        )
        self.sequence_starts, self.sequence_ends = seqs

    def __len__(self):
        """Number of samples in an epoch."""
        return floor(len(self.sequence_starts) * self.sample_rate)

    def __getitem__(self, index):
        """Return training sample."""
        if index > len(self):
            raise IndexError(
                "The training dataset is exhausted."
            )


        rem = index % self.sample_rate
        index = floor(index / self.sample_rate)

        offset = 0
        if self.augment and not self.validation:
            if self.sample_rate < 1.0:
                offset = self.rng.integers(int(1.0 / self.sample_rate))
                index = min(index + offset, len(self.sequence_starts) - 1)

        # We load a larger window when input is rotated to avoid
        # missing values.
        if self.augment:
            if not self.full:
                scene_size = int(1.42 * self.scene_size)
                rem = scene_size % self.max_scale
                if rem != 0:
                    scene_size += self.max_scale - rem
                ang = -180 + 360 * self.rng.random()
            else:
                scene_size = self.scene_size
                ang = None
            flip = self.rng.random() > 0.5
        else:
            scene_size = self.scene_size
            ang = None
            flip = False

        start_index = self.sequence_starts[index]
        if self.sample_rate > 1:
            max_len = self.sequence_ends[index] - self.sequence_starts[index]
            start_index = self.sequence_starts[index] + floor(rem / (self.sample_rate - 1) * max_len)

        if not self.full:
            # Find valid input range for last sample in sequence
            start_index = self.sequence_starts[index]
            ref_start = start_index
            if not self.include_input_steps:
                ref_start += self.sequence_length
            ref_end = start_index + self.sequence_length + self.forecast

            ref_offset = np.where(self.valid_ref[ref_start:ref_end])[0][-1]
            ref_index = ref_start + ref_offset
            rd_ind = np.where(self.reference_files[ref_index])[0][0]
            slices = self.reference_datasets[rd_ind].find_random_scene(
                self.reference_files[ref_index][rd_ind],
                self.rng,
                multiple=4,
                scene_size=scene_size,
                quality_threshold=self.quality_threshold[rd_ind],
            )
            if slices is None:
                LOGGER.warning(
                    " Couldn't find a scene in reference file '%s' satisfying "
                    "the quality requirements. Falling back to another "
                    "radomly-chosen sample.",
                    self.reference_files[ref_index][0]
                )
                new_ind = self.rng.integers(0, len(self))
                return self[new_ind]
        else:
            slices = (0, scene_size[0], 0, scene_size[1])

        x = {}
        y = {}

        any_ref_data = False

        for step in range(self.sequence_length):
            step_index = start_index + step
            if step < self.sequence_length:

                try:
                    x_i = self.load_input_sample(
                        self.input_files[step_index], slices, self.scene_size, rotate=ang, flip=flip
                    )
                except Exception:
                    LOGGER.warning(
                        "Encountered an error when loading input data from files '%s'."
                        "Falling back to another radomly-chosen sample.",
                        self.input_files[step_index]
                    )
                    new_ind = self.rng.integers(0, len(self))
                    return self[new_ind]

                for name, inpt in x_i.items():
                    x.setdefault(name, []).append(inpt)
                if self.include_input_steps:

                    try:
                        y_i = self.load_reference_sample(
                            self.reference_files[step_index], slices, self.scene_size, rotate=ang, flip=flip
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "Encountered an error when loading reference data from files '%s'."
                            "Falling back to another radomly-chosen sample.",
                            self.reference_files[step_index]
                        )
                        new_ind = self.rng.integers(0, len(self))
                        return self[new_ind]

                    if self.shrink_output:
                        y_i = {
                            name: center_crop(
                                tensor, tensor.shape[-1] // self.shrink_output
                            ) for name, tensor in y_i.items()
                        }
                    for name, inpt in y_i.items():
                        if torch.any(torch.isfinite(inpt)):
                            any_ref_data = True
                        y.setdefault(name, []).append(inpt)

        if self.forecast == 0:
            return x, y

        forecast_steps = np.arange(0, self.forecast_range)
        if self.forecast_range > self.forecast:
            forecast_steps = self.rng.permutation(forecast_steps)

        for step in forecast_steps[:self.forecast]:
            step_index = start_index + self.sequence_length + step

            try:
                y_i = self.load_reference_sample(
                    self.reference_files[step_index], slices, self.scene_size, rotate=ang, flip=flip
                )
            except Exception:
                LOGGER.warning(
                    "Encountered an error when loading reference data from files '%s'."
                    "Falling back to another radomly-chosen sample.",
                    self.reference_files[step_index]
                )
                new_ind = self.rng.integers(0, len(self))
                return self[new_ind]

            for name, inpt in y_i.items():
                if torch.any(torch.isfinite(inpt)):
                    any_ref_data = True
                y.setdefault(name, []).append(inpt)

            lead_time = self.time_step * (1 + step_index - start_index - self.sequence_length)
            minutes = lead_time.astype("int64") // 60
            x.setdefault("lead_time", []).append(minutes)

        # If there's no reference data, return other sample.
        if not any_ref_data:
            LOGGER.warning(
                "No valid reference data for sequence input starting at "
                "%s. Falling back to another radomly-chosen sample.",
                self.times[start_index]
            )
            new_ind = self.rng.integers(0, len(self))
            return self[new_ind]


        x["lead_time"] = torch.tensor(x["lead_time"])

        return x, y
