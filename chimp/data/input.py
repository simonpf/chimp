"""
chimp.data.input
===============

This sub-module defines classes for representing different input types.
"""
from dataclasses import dataclass
from functools import cache
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import xarray as xr

from pytorch_retrieve.tensors import MaskedTensor

from chimp import extensions
from chimp.utils import get_date
from chimp.data.utils import scale_slices
from chimp.data.source import DataSource


LOGGER = logging.getLogger(__name__)




class InputBase(DataSource):
    """
    Base class for all inputs that keeps track of all instances.
    """

    ALL_INPUT_DATASETS = {}

    def __init__(self, dataset_name, input_name):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name
        self.input_name = input_name
        self.ALL_INPUT_DATASETS[dataset_name] = self
        self.stack = False
        self.stacking_mode = "before"

    @classmethod
    def register_dataset(cls, name: str, dataset: DataSource) -> None:
        """
        Register a given dataset a input.

        Args:
            name: The name of the input dataset.
        """
        cls.ALL_INPUT_DATASETS[name] = dataset

    @classmethod
    def get_input_dataset(cls, name: str) -> "InputBase":
        """
        Get an input object by its name.

        Return:
            The input object of the given name if it exists.

        Raises:
            ValueError if there is no input with the given name
        """
        name = name.lower()
        if not name in cls.ALL_INPUT_DATASETS:
            raise ValueError(
                f"Input '{name}' is not a known input datasets. Currently known input "
                f"datasets are: {list(cls.ALL_INPUT_DATASETS.keys())}"
            )
        return cls.ALL_INPUT_DATASETS[name]


def get_input_dataset(name: Union[str, InputBase]) -> InputBase:
    """
    Get an input object by its name.

    Raises:
        ValueError if there is no input with the given name
    """
    from . import seviri
    from . import gpm
    from . import gridsat
    from . import ssmi
    from . import patmosx
    from . import wxfm
    from . import dem

    extensions.load()

    if isinstance(name, DataSource):
        return name
    return InputBase.get_input_dataset(name)


def get_input_datasets(input_list: List[Union[str, InputBase]]) -> List[InputBase]:
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
    return [get_input_dataset(inpt) for inpt in input_list]


@dataclass
class InputDataset(InputBase):
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
        dataset_name: str,
        input_name: str,
        scale: int,
        variables: Union[str, List[str]],
        n_dim: int = 2,
        spatial_dims: Tuple[str, str] = ("y", "x"),
    ):
        """
        Args:
            dataset_name: The name of the dataset that uniquely identifies this
                specific dataset.
            input_name: The (not necessarily uniqye) name of the input data provided
                by the input datasets.
            scale: The scale of the input.
            variables: List of the variables to load from each input file.
            n_dim: The number of dimensions in the input.
            spatial_dims: The name of the spatial dimensions of the data.
        """
        InputBase.__init__(self, dataset_name, input_name)
        self.dataset_name = dataset_name
        self.scale = scale
        self.variables = variables
        self.n_dim = n_dim
        self.spatial_dims = spatial_dims[:self.n_dim]

    def __hash__(self):
        """
        Use input dataset name to hash input dataset.
        """
        return hash(self.name)

    def get_files_to_stack(
            self,
            base_file: Path,
            stack: int,
            mode: str,
            time_step: np.timedelta64
    ) -> List[Path]:
        """
        Determine input files to stack.

        Args:
            base_file: The original input file.
            stack: The number of input files to stack.
            mode: The stacking mode ('before', 'after', 'centered')

        Return:
            A list containing the input files corresponding to the files to stack.
        """
        base_file = Path(base_file)
        date = get_date(base_file)
        stem = "_".join(base_file.name.split("_")[:-3])
        if mode.lower() == "before":
            deltas = np.arange(-stack + 1, 1)
        elif mode.lower() == "after":
            deltas = np.arange(0, stack)
        else:
            deltas = np.arange(-(stack // 2), (stack // 2 + stack % 2))
        times = [(date + time_step * delta).astype("datetime64[s]").item() for delta in deltas]
        files = [base_file.parent / time.strftime(f"{stem}_%Y%m%d_%H_%M.nc") for time in times]
        return [path if path.exists else None for path in files]

    def load_sample(
        self,
        input_file: Path,
        crop_size: Union[int, Tuple[int, int]],
        base_scale: int,
        slices: Tuple[slice, slice],
        rng: np.random.Generator,
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
            slices: Tuple of slices defining the part of the data to load.
            rng: A numpy random generator object to use to generate random
                data.
            rotate: If given, the should specify the number of degree by
                which the input should be rotated.
            flip: Bool indicated whether or not to flip the input along the
                last dimensions.

        Return:
            A torch tensor containing the loaded input data.
        """
        rel_scale = self.scale / base_scale

        if isinstance(crop_size, int):
            crop_size = (crop_size,) * self.n_dim
        crop_size = tuple((int(size / rel_scale) for size in crop_size))

        if input_file != "None":
            try:
                data = xr.open_dataset(input_file, decode_timedelta=False, chunks=None, cache=False)
                data.close()
                del data
            except OSError:
                LOGGER.warning(
                    "Reading of the input file '%s' failed. Skipping.",
                    input_file
                )
                input_file = "None"


        if input_file != "None":
            slices = scale_slices(slices, rel_scale)

            try:
                with xr.open_dataset(input_file, decode_timedelta=False, chunks=None, cache=False) as data:
                    vars = self.variables
                    if not isinstance(vars, list):
                        vars = [vars]
                    all_data = []
                    for vrbl in vars:

                        if slices is None:
                            slices = (slice(0, crop_sizes[0]), slice(0, crop_sizes[1]))

                        if slices[0].stop is not None:
                            height = slices[0].stop - slices[0].start
                            width = slices[1].stop - slices[1].start
                        else:
                            height = crop_size[0]
                            width = crop_size[1]

                        x_s = np.nan * np.zeros((height, width) + data[vrbl].shape[2:], dtype=np.float32)
                        slices_offset = slices

                        if "row_start" in data.attrs or "row_start" in data:
                            if "row_start" in data.attrs:
                                row_start = data.attrs["row_start"]
                                col_start = data.attrs["col_start"]
                            else:
                                row_start = data["row_start"].data
                                col_start = data["col_start"].data

                            slices_offset = (
                                slice(max(slices[0].start - row_start, 0), max(slices[0].stop - row_start, 0)),
                                slice(max(slices[1].start - col_start, 0), max(slices[1].stop - col_start, 0)),
                            )
                            height = min(data[vrbl].shape[0], slices_offset[0].stop) - slices_offset[0].start
                            width = min(data[vrbl].shape[1], slices_offset[1].stop) - slices_offset[1].start

                        x_s[:height, :width] = data[vrbl][dict(zip(self.spatial_dims, slices_offset))].data

                        if np.issubdtype(x_s.dtype, np.timedelta64):
                            x_s = x_s.astype("timedelta64[m]").astype("float32")
                            x_s[x_s < -1e16] = np.nan
                        if x_s.ndim < 3:
                            x_s = x_s[..., None]
                        if x_s.ndim > 3:
                            x_s = x_s.reshape(x_s.shape[:2] + (-1,))
                        x_s = np.transpose(x_s, (2, 0, 1))
                        all_data.append(x_s)
                    x_s = np.concatenate(all_data, axis=0)
                data.close()
                del data

                # Apply augmentations
                x_s = self.transform(x_s, crop_size, rotate, flip)
            except Exception as exc:
                raise exc
                x_s = np.nan * np.ones(((self.n_channels,) + crop_size), dtype=np.float32)
        else:
            x_s = np.nan * np.ones(((self.n_channels,) + crop_size), dtype=np.float32)


        x_s = torch.tensor(x_s.copy(), dtype=torch.float32)

        return x_s

    def load_data(
        self,
        input_file: Path,
    ) -> torch.Tensor:
        """
        Load input data sample from file.

        Args:
            input_file: Path pointing to the input file from which to load the
                data.

        Return:
            A torch tensor containing the loaded input data.
        """
        with xr.open_dataset(input_file, chunks=None, cache=False) as data:
            vars = self.variables
            if not isinstance(vars, list):
                vars = [vars]
            all_data = []
            for vrbl in vars:
                x_s = data[vrbl].data
                if x_s.ndim < 3:
                    x_s = x_s[None]
                x_s = np.transpose(x_s, (2, 0, 1))
                if np.issubdtype(x_s.dtype, np.floating):
                    x_s = x_s.astype(np.float32)
                all_data.append(x_s)
            x_s = torch.tensor(np.concatenate(all_data, axis=0))

        return x_s

    def find_random_scene(
            self,
            path,
            rng,
            multiple=4,
            scene_size=256,
            valid_fraction=0.2
    ):
        """
        Finds a random crop in the input data that is guaranteeed to have
        valid observations.

        Args:
            path: The path of the reference data file.
            rng: A numpy random generator instance to randomize the scene search.
            multiple: Limits the scene position to coordinates that are multiples
                of the given value.
            valid_fraction: The minimum amount of valid samples in the
                region.

        Return:
            A tuple ``(i_start, i_end, j_start, j_end)`` defining the position
            of the random crop.
        """
        with xr.open_dataset(path, chunks=None, cache=False) as data:
            if "latitude" in data.dims:
                n_rows = data.latitude.size
                n_cols = data.longitude.size
            else:
                n_rows = data.y.size
                n_cols = data.x.size

            if "center_indices" in data.dims:
                row_inds = data.row_inds_swath_center.data
                col_inds = data.col_inds_swath_center.data

                valid = (
                    (row_inds > scene_size // 2)
                    * (row_inds < n_rows - scene_size // 2)
                    * (col_inds > scene_size // 2)
                    * (col_inds < n_cols - scene_size // 2)
                )

                if valid.sum() == 0:
                    return None

                row_inds = row_inds[valid]
                col_inds = col_inds[valid]

                ind = rng.choice(np.arange(valid.sum()))
                row_c = row_inds[ind]
                col_c = col_inds[ind]

                i_start = int((row_c - scene_size // 2) // multiple * multiple)
                i_end = int(i_start + scene_size)
                j_start = int((col_c - scene_size // 2) // multiple * multiple)
                j_end = int(j_start + scene_size)

            else:

                input_data = []
                vars = self.variables
                if not isinstance(vars, list):
                    vars = [vars]
                for vrbl in vars:
                    x_s = data[vrbl].data
                    if np.issubdtype(x_s.dtype, np.timedelta64):
                        x_s = x_s.astype("timedelta64[m]").astype("float32")
                        x_s[x_s < -1e16] = np.nan
                    if x_s.ndim < 3:
                        x_s = x_s[..., None]
                    if x_s.ndim > 3:
                        x_s = x_s.reshape(x_s.shape[:2] + (-1,))
                    x_s = np.transpose(x_s, (2, 0, 1))
                    input_data.append(x_s)

                input_data = np.concatenate(input_data, axis=0)
                valid = np.any(np.isfinite(input_data), axis=0)

                found = False
                cnt = 0
                while not found:

                    if cnt > 20:
                        return None
                    cnt += 1
                    i_start = rng.integers(0, (n_rows - scene_size) // multiple)
                    i_end = i_start + scene_size // multiple
                    j_start = rng.integers(0, (n_cols - scene_size) // multiple)
                    j_end = j_start + scene_size // multiple

                    i_start = int(i_start * multiple)
                    i_end = int(i_end * multiple)
                    j_start = int(j_start * multiple)
                    j_end = int(j_end * multiple)

                    if valid[i_start:i_end, j_start:j_end].mean() > valid_fraction:
                        found = True

        return (i_start, i_end, j_start, j_end)


class StaticInputDataset(InputDataset):
    """
    Class to identify static input data.
    """
    pass



def get_input_map(
        inputs: Dict[str, torch.Tensor],
        ref_shape: Optional[Tuple[int, int]] = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Calculate input map at lowest input scale.

    Args:
        inputs: A dictionary holding the retrieval inputs.
        ref_shape: An optional spatial shape to which to upsample the input masks.

    Return:
        If the retrieval inputs correspond to a single time step, a 4D torch tensor is returned
        holding binary maps for all inputs in 'inputs' stacked along the first dimension.
        If the retrieval inputs contain a sequence of input tensors, a list of 4D input maps for
        each input steps is returned.
    """
    if ref_shape is None:
        max_dim = None
        ref_name = None
        ref_shape = None
        for name, tensor in inputs.items():

            if isinstance(tensor, list):
                tensor = tensor[0]

            if max_dim is None:
                max_dim = max(tensor.shape[-2:])
                ref_name = name
                ref_shape = tuple(tensor.shape[-2:])
            else:
                dim = max(tensor.shape[-2:])
                if dim > max_dim:
                    max_dim = dim
                    ref_name = name
                    ref_shape = tuple(tensor.shape[-2:])

    if isinstance(inputs[ref_name], list):
        seq_len = len(inputs[ref_name])
    elif inputs[ref_name].dim() > 4:
        seq_len = inputs[ref_name].shape[2]
    else:
        seq_len = None

    if seq_len is None:
        input_maps = []
        for name, tensor in inputs.items():
            if tensor.ndim < 4:
                tensor = tensor[None]
            dim = tensor.shape[-2:]
            up_fac = (ref_shape[0] / dim[0], ref_shape[1] / dim[1])
            upsample = nn.Upsample(scale_factor=up_fac)
            input_map = upsample(
                tensor.isfinite().any(dim=1)[:, None].to(dtype=torch.float32)
            )
            input_maps.append(input_map > 0.0)
        input_map = torch.cat(input_maps, 1)
        return input_map

    input_maps = [[] for _ in range(seq_len)]
    for name, tensors in inputs.items():
        if name.endswith("_mask") or name.endswith("_time"):
            continue
        if isinstance(tensors, torch.Tensor):
            tensors = torch.unbind(tensors, dim=2)
        for step, tensor in enumerate(tensors):
            if tensor.ndim < 4:
                tensor = tensor[None]
            dim = tensor.shape[-2:]
            up_fac = (ref_shape[0] / dim[0], ref_shape[1] / dim[1])
            upsample = nn.Upsample(scale_factor=up_fac)
            input_map = upsample(
                tensor.isfinite().any(dim=1)[:, None].to(dtype=torch.float32)
            )
            input_maps[step].append(input_map > 0.0)
    input_maps = [torch.cat(input_map, 1) for input_map in input_maps]
    return input_maps




def get_input_age(
        inputs: Dict[str, List[torch.Tensor]],
        bidirectional: bool = True,
        ref_shape: Optional[Tuple[int, int]] = None
) -> List[torch.Tensor]:
    """
    Calculate the input age for for sequence inputs.

    Args:
        inputs: A dictionary holding the input sequences for all inputs.
        bidirectional: If True, the age will correspond to the signed, shortest step-difference
            to nearest input step. Else only the nearest distance in forward direction will
            be considered and values will be positive.
        ref_shape: An optional spatial shape to which to upsample the input masks.

    Return:
        A list of 4D tensors containing the respective age of all observations for all time
        steps in the sequence.
    """
    if ref_shape is None:
        max_dim = None
        ref_name = None
        for name, seq in inputs.items():
            if max_dim is None:
                max_dim = max(seq[0].shape[-2:])
                ref_name = name
            else:
                dim = max(seq[0].shape[-2:])
                if dim > max_dim:
                    max_dim = dim
                    ref_name = name
        ref_shape = tuple(inputs[ref_name][0].shape[-2:])

    seq = next(iter(inputs.values()))
    n_batch = seq[0].shape[0]
    device = seq[0].device
    dtype = seq[0].dtype
    seq_len = len(seq)
    map_shape = (n_batch, len(inputs)) + ref_shape

    input_maps = []
    input_names = []
    ages = []

    curr_age = torch.nan * torch.zeros(map_shape, dtype=dtype, device=device)
    for step in range(seq_len):
        ages.append(np.nan * torch.zeros(map_shape, dtype=dtype, device=device))
        for ind, (name, seq) in enumerate(inputs.items()):
            tensor = seq[step]

            if tensor.ndim < 4:
                tensor = tensor[None]

            dim = tensor.shape[-2:]
            up_fac = (ref_shape[0] / dim[0], ref_shape[1] / dim[1])
            upsample = nn.Upsample(scale_factor=up_fac)
            input_map = upsample(
                tensor.isfinite().any(dim=1)[:, None].to(dtype=torch.float32)
            )
            input_map = input_map > 0
            curr_age[:, ind][~input_map[:, 0]] += 1.0
            curr_age[:, ind][input_map[:, 0]] = 0.0

        update_mask = (curr_age.abs() < ages[-1].abs()) + torch.isnan(ages[-1])
        ages[-1][update_mask] = curr_age[update_mask]

    if bidirectional:
        curr_age = torch.nan * torch.zeros(map_shape, dtype=dtype, device=device)
        for step in range(seq_len - 1, -1, -1):
            for ind, (name, seq) in enumerate(inputs.items()):
                tensor = seq[step]

                if tensor.ndim < 4:
                    tensor = tensor[None]

                dim = tensor.shape[-2:]
                up_fac = (ref_shape[0] / dim[0], ref_shape[1] / dim[1])
                upsample = nn.Upsample(scale_factor=up_fac)
                input_map = upsample(
                    tensor.isfinite().any(dim=1)[:, None].to(dtype=torch.float32)
                )
                input_map = input_map > 0
                curr_age[:, ind][~input_map[:, 0]] -= 1.0
                curr_age[:, ind][input_map[:, 0]] = 0.0

            update_mask = (curr_age.abs() < ages[step].abs()) + torch.isnan(ages[step])
            ages[step][update_mask] = curr_age[update_mask]

    return ages



class InputLoader():
    """
    The InputLoader class loads CHIMP input data for the operational
    application of CHIMP retrievals.
    """
    def __init__(
            self,
            path: Union[Path, List[Path]],
            input_datasets: List[str],
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None,
            missing_value_policy: str = "sparse",
            time_step: Optional[np.timedelta64] = None,
    ):
        """
        Args:
            path: The path pointing to the directory containing the inputs
                or pathes to a set of files.
            input_datasets: A list of names of input datasets.
            start_time: An optional start time to limit the input samples loaded
                by the loader.
            end_time: An optional end time to limit the input samples loaded
                by the loader.
            missing_value_policy: The name of the policy defining how to handle
               missin values.
            time_step: The time step between consecutive inputs.
        """
        self.path = path
        self.input_datasets = [
            get_input_dataset(input_dataset) for input_dataset in input_datasets
        ]

        self.missing_value_policy = missing_value_policy

        n_input_datasets = len(self.input_datasets)
        sample_files = {}
        scene_sizes = [None] * n_input_datasets

        for input_ind, input_dataset in enumerate(self.input_datasets):
            if isinstance(input_dataset, StaticInputDataset):
                continue
            times, input_files = input_dataset.find_training_files(self.path)
            # Determine input size for all inputs.
            if len(input_files) > 0:
                with xr.open_dataset(input_files[0], decode_timedelta=False, chunks=None, cache=False) as scene:
                    scene_sizes[input_ind] = [
                        scene[dim].size for dim in input_dataset.spatial_dims
                    ]
            for time, input_file in zip(times, input_files):
                files = sample_files.setdefault(time, ([None] * n_input_datasets))
                files[input_ind] = input_file

        sample_files_filtered = {}
        for time, files in sample_files.items():
            if start_time is not None:
                if time < start_time:
                    continue
            if end_time is not None:
                if time > end_time:
                    continue
            sample_files_filtered[time] = [str(fle) for fle in files]

        self.sample_files = sample_files_filtered

        self.scene_sizes = scene_sizes

        self.times = np.array(list(self.sample_files.keys()))
        if time_step is None:
            times = np.array(list(self.sample_files.keys()))
            if len(times) <= 1:
                time_step = None
            else:
                time_step = np.diff(np.sort(times)).min()
        self.time_step = time_step

        # Determine static input files
        for input_ind, input_dataset in enumerate(self.input_datasets):
            if not isinstance(input_dataset, StaticInputDataset):
                continue
            times, input_files = input_dataset.find_training_files(self.path, self.times)
            with xr.open_dataset(input_files[0], decode_timedelta=False, chunks=None, cache=False) as scene:
                scene_sizes[input_ind] = [
                    scene[dim].size for dim in input_dataset.spatial_dims
                ]
            for time, input_file in zip(times, input_files):
                self.sample_files[time][input_ind] = input_file


        self.rng = np.random.default_rng()
        self.dtype = times[0].dtype


    def __iter__(self):
        for time in self.times:
            yield time, self.get_input(time)

    def __len__(self):
        """
        The number of input samples.
        """
        return len(self.sample_files)


    def get_input(self, time: np.datetime64) -> Dict[str, torch.Tensor]:
        """
        Get input for a given time.

        Args:
            time: Time stamp defining the time for which to retrieve  inputs.

        Return:
            A dictionary containing the input tensors from all input datasets.
        """
        time = time.astype(self.dtype)

        if not time in self.sample_files:
            raise RuntimeError(
                "No input for time '%s' available.",
                time
            )

        files = self.sample_files[time]

        inputs = {}
        for ind, input_dataset in enumerate(self.input_datasets):
            x = input_dataset.load_sample(
                files[ind], self.scene_sizes[ind], input_dataset.scale, None,
                None,
            )
            inputs[input_dataset.input_name] = x[None]

        return inputs


class SequenceInputLoader(InputLoader):
    """
    An input loader for sequences of input observations.
    """
    def __init__(
            self,
            path: Union[Path, List[Path]],
            input_datasets: List[str],
            sequence_length: int,
            forecast: int = 0,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None,
            time_step: Optional[np.timedelta64] = None,
            temporal_overlap: Optional[int] = None
    ):
        """
        Args:
            path: The path pointing to the directory containing the inputs
                or pathes to a set of files.
            input_datasets: A list of names of input datasets.
            sequence_length: The length of the input sequences.
            forecast: The number of forecast steps to perform.
            start_time: An optional start time to limit the input samples loaded
                by the loader.
            end_time: An optional end time to limit the input samples loaded
                by the loader.
            time_step: The time step between consecutive inputs.
            temporal_overlap: The amount overlapping time steps between
                consectutive retrievals.
        """
        super().__init__(
            path=path,
            input_datasets=input_datasets,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step
        )
        self.sequence_length = sequence_length
        self.forecast = forecast
        if temporal_overlap is None:
            temporal_overlap = self.sequence_length // 2
        if temporal_overlap >= sequence_length:
            raise ValueError(
                "Temporal overlap must not exceed the sequence length."
            )
        self.temporal_overlap = temporal_overlap

    def __iter__(self):
        offset = (self.sequence_length - 1) * self.time_step
        end_time = self.times.max()

        curr_time = min(self.times.min() + offset, end_time)

        while curr_time <= end_time:
            try:
                yield curr_time, self.get_input(curr_time)
            except RuntimeError:
                LOGGER.warning("Found no input for step %s.", curr_time)
            curr_time += (self.sequence_length - self.temporal_overlap) * self.time_step

        if curr_time < end_time:
            try:
                yield end_time, self.get_input(end_time)
            except RuntimeError:
                LOGGER.warning("Found no input for step %s.", end_time)


    def get_input(self, time: np.datetime64) -> Dict[str, torch.Tensor]:
        """
        Get input for a given time.

        Args:
            time: Time stamp defining the time for which to retrieve  inputs.

        Return:
            A dictionary containing the input tensors from all input datasets.
        """
        sequence_times = np.flip(
            time - np.arange(self.sequence_length) * self.time_step
        )
        sequence_times = sequence_times.astype(self.dtype)

        any_input = any([time in self.sample_files for time in sequence_times])
        if not any_input:
            raise RuntimeError(
                "No sequence input available in the range '%s' - '%s'.",
                sequence_times[0],
                sequence_times[-1]
            )

        inputs = {}
        for time in sequence_times:
            files = self.sample_files.get(time, ["None"] * len(self.input_datasets))
            for ind, input_dataset in enumerate(self.input_datasets):
                x = input_dataset.load_sample(
                    input_file=files[ind],
                    crop_size=self.scene_sizes[ind],
                    base_scale=input_dataset.scale,
                    slices=None,
                    rng=self.rng
                )
                inputs.setdefault(input_dataset.input_name, []).append(x[None])


        if self.forecast > 0:
            lead_time = self.time_step * (np.arange(self.forecast) + 1)
            minutes = lead_time.astype("timedelta64[m]").astype("int64")
            inputs["lead_time"] = torch.tensor(minutes)[None]

        return inputs


class SparseSequenceInputLoader(InputLoader):
    """
    An input loader for sequences of input observations.
    """
    def __init__(
            self,
            path: Union[Path, List[Path]],
            input_datasets: List[str],
            sequence_length: int,
            forecast: int = 0,
            start_time: Optional[np.datetime64] = None,
            end_time: Optional[np.datetime64] = None,
            time_step: Optional[np.timedelta64] = None,
            temporal_overlap: Optional[int] = None
    ):
        """
        Args:
            path: The path pointing to the directory containing the inputs
                or pathes to a set of files.
            input_datasets: A list of names of input datasets.
            sequence_length: The length of the input sequences.
            forecast: The number of forecast steps to perform.
            start_time: An optional start time to limit the input samples loaded
                by the loader.
            end_time: An optional end time to limit the input samples loaded
                by the loader.
            time_step: The time step between consecutive inputs.
            temporal_overlap: The amount overlapping time steps between
                consectutive retrievals.
        """
        super().__init__(
            path=path,
            input_datasets=input_datasets,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step
        )
        self.sequence_length = sequence_length
        self.forecast = forecast
        if temporal_overlap is None:
            temporal_overlap = self.sequence_length // 2
        if temporal_overlap >= sequence_length:
            raise ValueError(
                "Temporal overlap must not exceed the sequence length."
            )
        self.temporal_overlap = temporal_overlap
        self.output_scale = 4

    @cache
    def get_times_and_files(self, dataset: [InputDataset]) -> np.ndarray:
        """
        Get available input time for a given input dataset.
        """
        times, files = dataset.find_training_files(self.path, times=None)
        files = np.array([str(path) for path in files])
        mask = np.ones_like(times, dtype=bool)
        times = times[mask]
        files = files[mask]
        return times, files

    @cache
    def get_files_in_range(
            self,
            dataset: [InputDataset],
            start_time: np.datetime64,
            end_time: np.datetime64
    ) -> np.ndarray:
        """
        Get available input time for a given input dataset.
        """
        times, files = self.get_times_and_files(dataset)
        if len(times) == 0:
            return np.zeros(0)
        mask = (start_time <= times) * (times <= end_time)
        times = times[mask]
        files = files[mask]
        return files

    def __iter__(self):
        curr_start_time = self.times.min()
        curr_end_time = curr_start_time + self.time_step * self.sequence_length
        end_time = self.times.max()
        while curr_start_time <= end_time:
            try:
                yield curr_end_time, self.get_input(curr_start_time, curr_end_time)
            except RuntimeError:
                LOGGER.warning(
                    "Found no input for step %s.",
                    curr_start_time
                )
            curr_start_time += self.time_step * (self.sequence_length - self.temporal_overlap)
            curr_end_time += self.time_step * (self.sequence_length - self.temporal_overlap)

    def get_input(self, start_time: np.datetime64, end_time: np.datetime64) -> Dict[str, torch.Tensor]:
        """
        Get input for a given time.

        Args:
            time: Time stamp defining the time for which to retrieve  inputs.

        Return:
            A dictionary containing the input tensors from all input datasets.
        """

        inputs = {}
        # Load input data
        any_inpt = False
        valid_inputs = 0

        for ds_ind, input_dataset in enumerate(self.input_datasets):

            files = self.get_files_in_range(input_dataset, start_time, end_time)[:2]
            if files.size > 0:
                files = self.rng.permutation(files)

            if hasattr(input_dataset, "input_length"):
                input_length = input_dataset.input_length
            else:
                input_length = self.sequence_length

            step = 0
            while step < input_length:

                if step < len(files):
                    input_file = files[step]
                    if hasattr(input_dataset, "input_drop"):
                        if self.rng.random() < input_dataset.input_drop:
                            input_file = "None"
                else:
                    input_file = "None"

                try:
                    input_data = input_dataset.load_sample(
                        input_file, self.scene_sizes[ds_ind], input_dataset.scale, None, None
                    )
                except Exception as exc:
                    raise exc
                    LOGGER.warning(
                        "Loading of input data from input file '%s' failed.",
                        input_file
                    )
                    continue

                if torch.isfinite(input_data).any():
                    any_inpt = True

                inputs.setdefault(input_dataset.input_name, []).append(
                    input_data
                )
                input_time = input_dataset.load_sample_time(
                        input_file, self.scene_sizes[ds_ind], input_dataset.scale, end_time, None
                )
                inputs.setdefault(input_dataset.input_name + "_time", []).append(
                    input_time
                )

                step += 1


        if not any_inpt:
            raise ValueError(
                "No valid input for retrieval time %s - %s.",
                start_time,
                end_time
            )

        inputs = {name: torch.stack(tnsrs, 1)[None] for name, tnsrs in inputs.items()}

        target_times = np.arange(start_time, end_time + self.time_step, self.time_step)
        scene_size = self.scene_sizes[ds_ind]
        scaling = self.output_scale // input_dataset.scale
        scene_size = (scene_size[0] // scaling, scene_size[1] // scaling)
        target_times = [(time - end_time).astype("timedelta64[m]").astype(np.float32) for time in target_times]
        targets = ["surface_precip", "rain_water_content", "snow_water_content"]
        for targ in targets:
            inputs[targ + "_target_time"] = torch.stack(
                [torch.ones((1, 1,) + scene_size) * time for time in target_times],
                dim=2
            )


        # Generate masks
        input_names = [name for name in inputs.keys() if not name.endswith("_time")]
        for name in input_names:
            tnsr = inputs[name]
            mask = torch.isnan(tnsr).all(-1).all(-1).all(1)
            inputs[name + "_mask"] = mask

        return inputs
