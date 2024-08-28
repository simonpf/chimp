"""
chimp.data.gpm
==============

This module implements functions the handling of data from the
GPM constellation.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict,  List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pansat
from pansat import Geometry, FileRecord
from pansat.granule import merge_granules
from pansat.time import TimeRange
from pansat.products.satellite.gpm import (
    l1c_r_gpm_gmi,
    l1c_r_gpm_gmi_b,
    l1c_metopa_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_npp_atms,
    l1c_noaa20_atms,
    l1c_f16_ssmis,
    l1c_f17_ssmis,
    l1c_f18_ssmis,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07b,
    l1c_xcal2021v_f18_ssmis_v07b,
    l1c_gcomw1_amsr2,
    l2b_gpm_cmb,
    l2b_gpm_cmb_b,
    l2b_gpm_cmb_c,
    l2a_gpm_dpr
)
from pyresample import AreaDefinition
import torch
import xarray as xr

from chimp.areas import Area
from chimp.data.input import InputDataset
from chimp.data.resample import resample_and_split
from chimp.data.reference import ReferenceDataset, RetrievalTarget
from chimp.data.mrms import MRMS_PRECIP_RATE
from chimp.data.utils import records_to_paths
from chimp.utils import get_date


from chimp.data.utils import get_output_filename, records_to_paths


LOGGER = logging.getLogger(__name__)


class GPML1CData(InputDataset):
    """
    Represents  input data derived from GPM L1C products.
    """
    def __init__(
        self,
        dataset_name: str,
        scale: int,
        products: List[pansat.Product],
        n_swaths: int,
        n_channels: int,
        radius_of_influence: float,
        input_name: Optional[str] = None,
        include_incidence_angle: bool = False,
        valid_time: np.timedelta64 = np.timedelta64(0, "ns")
    ):
        """
        Args:
            dataset_name: A name that uniquely identifies the input dataset.
            scale: The spatial scale to which the output will be mapped.
            products: The pansat products reprsenting the L1C products of the
                sensors.
            n_swaths: The number of swaths in the L1C files.
            n_channels: The number of channels in the input data.
            radius_of_influence: The radius of influence in meters to use for the
                resampling of the data.
            input_name: Optional input name in case it is meant to deviate from the
                dataset name.
            include_incidence_angle: Whether or not to include the incidence angle in
                the retrieval input.
        """
        if input_name is None:
            input_name = dataset_name
        target_names = ["tbs"]
        if include_incidence_angle:
            target_names = target_names + ["incidence_angle"]
        if valid_time > np.timedelta64(0, "ns"):
            target_names = target_names + ["age"]
        InputDataset.__init__(self, dataset_name, input_name, scale, target_names, n_dim=2)
        self.products = products
        self.n_swaths = n_swaths
        self.n_channels = (
            n_channels +
            include_incidence_angle +
            (valid_time > np.timedelta64(0, "ns"))
        )
        self.radius_of_influence = radius_of_influence
        self.include_incidence_angle = include_incidence_angle
        self.valid_time = valid_time


    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
        """
        Find input data files within a given file range from which to extract
        training data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [
                    path for path in all_files if prod.matches(path)
                ]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.find_files(TimeRange(start_time, end_time))
        return recs


    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given input data file.

        Args:
            path: A Path object pointing to the file to process.
            domain: An area object defining the training domain.
            output_folder: A path pointing to the folder to which to write
                the extracted training data.
            time_step: A timedelta object defining the retrieval time step.
        """
        path = records_to_paths(path)

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        input_data = self.products[0].open(path)
        swath_data =  []

        for swath in range(1, self.n_swaths + 1):
            new_names = {
                f"scan_time": "time",
            }
            if self.n_swaths > 1:
                new_names.update({
                    f"latitude_s{swath}": "latitude",
                    f"longitude_s{swath}": "longitude",
                    f"tbs_s{swath}": "tbs",
                    f"incidence_angle_s{swath}": "incidence_angle",
                    f"channels_s{swath}": "channels",
                })
            data = input_data.rename(new_names)

            vars = ["latitude", "longitude", "tbs", "incidence_angle", "time"]
            data = data[vars]
            if f"pixels_s{swath}" in data:
                data = data.rename({f"pixels_s{swath}": "pixels"})

            drop = []
            for var in data.variables.keys():
                for swath in range(1, self.n_swaths + 1):
                    if var.endswith(f"_s{swath}"):
                        drop.append(var)
            if not self.include_incidence_angle:
                drop.append('incidence_angle')
            data = data.drop(drop)

            # Need to expand scan time to full dimensions.
            time, _ = xr.broadcast(data.time, data.longitude)
            data["time"] = time
            tbs = data.tbs.data
            tbs[tbs < 0] = np.nan

            data_s = resample_and_split(
                    data,
                    domain[self.scale],
                    time_step,
                    self.radius_of_influence,
                    include_swath_center_coords=True
                )

            if data_s is None:
                return None

            if self.include_incidence_angle and data_s.incidence_angle.data.ndim == 2:
                angs, _ = xr.broadcast_to(data_s.incidence_angle, data_s.tbs)
                data_s["incidence_angle"] = angs

            swath_data.append(data_s)

        if len(swath_data) == 0:
            LOGGER.info(
                "Found no overpasses in file %s.",
                path.name
            )
            return None

        data = xr.concat(swath_data, "channels")
        if self.include_incidence_angle:
            angs = data.incidence_angle[{"channels": 0}].data
            data["incidence_angle"] = (("time", "y", "x",), angs)

        for time_ind  in range(data.time.size):

            data_t = data[{"time": time_ind}]

            tbs = data_t.tbs.data
            if np.isfinite(tbs).any(-1).sum() < 100:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    data_t.time.data
                )
                continue

            data_t["age"] = (("y", "x"), np.zeros(tbs.shape[:-1], dtype="timedelta64[ns]"))
            data_t["age"].data[:] = np.timedelta64("NAT")
            data_t.attrs["source_files"] = str(path.name)

            comp = {
                "dtype": "uint16",
                "scale_factor": 0.01,
                "zlib": True,
                "_FillValue": 2**16 - 1,
            }
            encoding = {
                "tbs": comp,
                "col_inds_swath_center": {"dtype": "int16"},
                "row_inds_swath_center": {"dtype": "int16"},
            }
            if self.include_incidence_angle:
                encoding["incidence_angle"] = {
                    "dtype": "int16",
                    "scale_factor": 0.1,
                    "zlib": True,
                    "_FillValue": 2**15 - 1,
                }

            age = np.timedelta64(0, "ns")
            while age <= self.valid_time:
                valid_mask = np.isfinite(tbs).any(-1)
                data_t["age"].data[valid_mask] = age
                filename = get_output_filename(
                    self.name, data.time[time_ind].data + age, time_step
                )
                output_file = output_folder / filename
                LOGGER.info(
                    "Writing training samples to %s.",
                    output_folder / filename
                )
                if output_file.exists():
                    output_data = xr.load_dataset(output_file)
                    mask = (output_data.age.data > data_t.age.data) * valid_mask
                    output_data.tbs.data[mask] = tbs[mask]
                    output_data.age.data[mask] = age
                    if self.include_incidence_angle:
                        output_data.incidence_angle.data[mask] = data_t.incidence_angle.data[mask]
                    output_data.to_netcdf(output_file, encoding=encoding)
                    output_data.attrs["source_files"] += f"\n{path.name}"
                else:
                    data_t.to_netcdf(output_folder / filename, encoding=encoding)
                age += time_step


GMI = GPML1CData("gmi", 4, [l1c_r_gpm_gmi, l1c_r_gpm_gmi_b], 2, 13, 15e3)
ATMS = GPML1CData("atms", 16, [l1c_noaa20_atms, l1c_npp_atms], 4, 9, 64e3)
ATMS_W_ANGLE = GPML1CData(
    "atms_w_angle",
    16,
    [l1c_noaa20_atms, l1c_npp_atms],
    4,
    9,
    64e3,
    input_name = "atms",
    include_incidence_angle=True
)
ATMS_2H = GPML1CData(
    "atms_2h",
    16,
    [l1c_noaa20_atms, l1c_npp_atms],
    4,
    9,
    64e3,
    input_name = "atms",
    include_incidence_angle=True,
    valid_time=np.timedelta64(120, "m")
)

MHS_PRODUCTS = [
    l1c_noaa18_mhs,
    l1c_noaa19_mhs,
    l1c_metopa_mhs,
    l1c_metopb_mhs,
    l1c_metopc_mhs,
]
MHS = GPML1CData("mhs", 8, MHS_PRODUCTS, 1, 5, 64e3)

SSMIS_PRODUCTS = [
    l1c_f16_ssmis,
    l1c_f17_ssmis,
    l1c_f18_ssmis,
    l1c_xcal2021v_f16_ssmis_v07b,
    l1c_xcal2021v_f17_ssmis_v07b,
    l1c_xcal2021v_f18_ssmis_v07b,
]
SSMIS = GPML1CData("ssmis", 8, SSMIS_PRODUCTS, 4, 11, 30e3)

AMSR2 = GPML1CData("amsr2", 4, [l1c_gcomw1_amsr2], 6, 12, 30e3)


#
# GPM CMB
#


class GPMCMB(ReferenceDataset):
    """
    Represents retrieval reference data derived from GPM combined radar-radiometer retrievals.
    """
    def __init__(self, name=None):
        if name is None:
            name = "cmb"
        self.name = name
        super().__init__(
            self.name,
            4,
            [RetrievalTarget("surface_precip")],
            quality_index=None
        )
        self.products = [l2b_gpm_cmb, l2b_gpm_cmb_b, l2b_gpm_cmb_c]
        self.scale = 4
        self.radius_of_influence = 6e3

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[FileRecord]:
        """
        Find input data files within a given file range from which to extract
        training data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [path for path in all_files if prod.matches(path)]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.find_files(TimeRange(start_time, end_time))
        return recs


    def find_random_scene(
            self,
            path,
            rng,
            multiple=4,
            scene_size=256,
            valid_fraction=0.2,
            **kwargs
    ):
        """
        Finds a random crop in the GPM CMB data that is guaranteeed to have
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
        try:
            with xr.open_dataset(path) as data:
                if "latitude" in data.dims:
                    n_rows = data.latitude.size
                    n_cols = data.longitude.size
                else:
                    n_rows = data.y.size
                    n_cols = data.x.size

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

                if rng is not None:
                    offset = rng.integers(
                        -min(j_start, scene_size // 4),
                        min(n_cols - j_end, scene_size // 4)
                    )
                    j_start += offset
                    j_end += offset

            return (i_start, i_end, j_start, j_end)
        except OSError:
            return None

    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given source file.

        Args:
           path: A Path object pointing to the file to process.
           domain: An area object defining the training domain.
           output_folder: A path pointing to the folder to which to write
               the extracted training data.
           time_step: A timedelta object defining the retrieval time step.
        """
        path = records_to_paths(path)
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        input_data = self.products[0].open(path)
        data = input_data.rename({
            "scan_time": "time",
            "estim_surf_precip_tot_rate": "surface_precip"
        })[["time", "surface_precip"]]

        surface_precip = data.surface_precip.data
        surface_precip[surface_precip < 0] = np.nan

        # Need to expand scan time to full dimensions.
        time, _ = xr.broadcast(data.time, data.longitude)
        data["time"] = time

        data = resample_and_split(
            data,
            domain[self.scale],
            time_step,
            radius_of_influence=self.radius_of_influence,
            include_swath_center_coords=True
        )

        if data is None:
            return None

        for time_ind  in range(data.time.size):

            data_t = data[{"time": time_ind}]

            precip = data_t.surface_precip.data
            invalid = precip < 0
            precip[invalid] = np.nan

            if np.isfinite(precip).sum() < 100:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    data_t.time.data
                )
                continue


            encoding = {
                "surface_precip": {"dtype": "float32", "zlib": True},
                "col_inds_swath_center": {"dtype": "int16"},
                "row_inds_swath_center": {"dtype": "int16"},
            }
            filename = get_output_filename(
                self.name, data.time[time_ind].data, time_step
            )

            LOGGER.info(
                "Writing training samples to %s.",
                output_folder / filename
            )
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


CMB = GPMCMB()


class GPMCMBAnd(GPMCMB):
    """
    This reference data class loads reference data from GPM CMB training files and
    another reference dataset.
    """
    def __init__(self, other: str):
        name = "cmb_and_" + other.name
        super().__init__(name=name)
        self.other = other


    def other_path(self, path):
        filename = path.name
        other_filename = self.other.name + filename[3:]
        other_path = path.parent.parent / self.other.name / other_filename
        return other_path

    def _find_training_files(
            self,
            path: Path,
            name: str
    ) -> Tuple[np.ndarray, List[Path]]:
        """
        Find training data files.

        Args:
            path: Path to the folder the training data for all input
                and reference datasets.
            name: The name of the dataset.

        Return:
            A tuple ``(times, files)`` specifying the timestamps and corresponding files
            the have been found for this input.
        """
        pattern = "*????????_??_??.nc"
        training_files = sorted(
            list((path / name).glob(pattern))
        )
        times = np.array(list(map(get_date, training_files)))
        return times, training_files

    def find_training_files(
            self,
            path: Path,
            times: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Path]]:
        """
        Find training data files.

        Args:
            path: Path to the folder the training data for all input
                and reference datasets.
            times: Optional array containing valid reference data times for static
                inputs.

        Return:
            A tuple ``(times, files)`` specifying the timestamps and corresponding files
            the have been found for this input.
        """
        cmb_times, cmb_files = self._find_training_files(path, "cmb")
        other_times, other_files = self._find_training_files(path, self.other.name)

        files = {cmb_time: cmb_file for cmb_time, cmb_file in zip(cmb_times, cmb_files)}
        other_files = {
            other_time: other_file for other_time, other_file in zip(other_times, other_files)
        }
        for time, other_file in other_files.items():
            if time not in files:
                files[time] = other_file
        return (np.array(list(files.keys())), list(files.values()))


    def find_random_scene(
        self,
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
        if not path.name.startswith("cmb"):
            return self.other.find_random_scene(
                path=path,
                rng=rng,
                multiple=multiple,
                scene_size=scene_size,
                quality_threshold=quality_threshold,
                valid_fraction=valid_fraction
            )

        if isinstance(scene_size, (int, float)):
            scene_size = (int(scene_size),) * 2

        try:
            with xr.open_dataset(path) as data:

                if self.quality_index is not None:
                    qi = data[self.quality_index].data
                else:
                    qi = np.isfinite(data[self.targets[0].name].data)

                other_path = self.other_path(path)
                if other_path.exists():
                    with xr.open_dataset(other_path) as other_data:
                        if self.other.quality_index is not None:
                            qi = np.maximum(qi, other_data[self.other.quality_index].data)
                        else:
                            qi = np.maximum(qi, np.isfinite(data[self.other.targets[0].name].data))

                found = False
                count = 0
                while not found:
                    if count > 20:
                        return None
                    count += 1
                    n_rows, n_cols = qi.shape
                    i_start = rng.integers(0, (n_rows - scene_size[0]) // multiple)
                    i_end = i_start + scene_size[0] // multiple
                    j_start = rng.integers(0, (n_cols - scene_size[1]) // multiple)
                    j_end = j_start + scene_size[1] // multiple

                    i_start = i_start * multiple
                    i_end = i_end * multiple
                    j_start = j_start * multiple
                    j_end = j_end * multiple

                    row_slice = slice(i_start, i_end)
                    col_slice = slice(j_start, j_end)

                    if (qi[row_slice, col_slice] > quality_threshold).mean() > valid_fraction:
                        found = True

            return (i_start, i_end, j_start, j_end)
        except Exception:
            LOGGER.warning(
                "Finding a random scene from reference file '%s' failed.",
                path
            )
            return None


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
        from pytorch_retrieve.tensors.masked_tensor import MaskedTensor

        if path is None:
            if isinstance(crop_size, tuple):
                n_rows, n_cols = crop_size
            else:
                n_rows = crop_size
                n_cols = crop_size
            return {
                target.name: MaskedTensor(
                    torch.nan * torch.zeros((n_rows, n_cols)),
                    mask = torch.ones((n_rows, n_cols), dtype=torch.bool)
                ) for target in self.targets
            }

        if not path.name.startswith("cmb"):
            other_targets = self.other.load_sample(
                path=path,
                crop_size=crop_size,
                base_scale=base_scale,
                slices=slices,
                rng=rng,
                rotate=rotate,
                flip=flip,
                quality_threshold=quality_threshold
            )
            for targ in self.targets:
                if not targ.name in other_targets:
                    shape = self.targets[name].shape + (crop_size,) * 2
                    other_targets[target] = torch.nan * torch.ones(shape)
            return other_targets


        targets = super().load_sample(
            path=path,
            crop_size=crop_size,
            base_scale=base_scale,
            slices=slices,
            rng=rng,
            rotate=rotate,
            flip=flip,
            quality_threshold=quality_threshold
        )

        other_path = self.other_path(path)
        if other_path.exists():
            other_targets = self.other.load_sample(
                path=other_path,
                crop_size=crop_size,
                base_scale=base_scale,
                slices=slices,
                rng=rng,
                rotate=rotate,
                flip=flip,
                quality_threshold=quality_threshold
            )
            for name in other_targets:
                if name in targets:
                    y_cmb = targets[name].base
                    y_other = other_targets[name]
                    mask = torch.isfinite(y_cmb)
                    y_other[mask] = y_cmb[mask].to(dtype=y_other.dtype)
                    targets[name] = y_other

        return targets


CMB_AND_MRMS = GPMCMBAnd(MRMS_PRECIP_RATE)

#
# GPM DPR
#


class GPMDPR(InputDataset):
    """
    Represents retrieval reference data derived from GPM combined radar-radiometer retrievals.
    """
    def __init__(
            self,
            bands: List[int] = [0, 1]
    ):
        if 0 in bands and 1 in bands:
            input_name = "dpr"
        elif 0 in bands:
            input_name = "dpr_ku"
        elif 1 in bands:
            input_name = "dpr_ka"
        else:
            raise RuntimeError(
                "The list of bands must be [0], [1], or [0, 1]."
            )

        super().__init__(
            input_name,
            input_name,
            4,
            "refls",
            n_dim=2
        )
        self.products = [l2a_gpm_dpr]
        self.scale = 4
        self.radius_of_influence = 6e3

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
        """
        Find input data files within a given file range from which to extract
        training data.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files to extract CHIMP training data from.
        """
        if path is not None:
            path = Path(path)
            all_files = sorted(list(path.glob("**/*.HDF5")))
            matching = []
            for prod in self.products:
                matching += [path for path in all_files if prod.matches(path)]
            return matching

        recs = []
        for prod in self.products:
            recs += prod.get(TimeRange(start_time, end_time))
        return [rec.local_path for rec in recs]

    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ):
        """
        Extract training samples from a given source file.

        Args:
           path: A Path object pointing to the file to process.
           domain: An area object defining the training domain.
           output_folder: A path pointing to the folder to which to write
               the extracted training data.
           time_step: A timedelta object defining the retrieval time step.
        """
        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)

        input_data = self.products[0].open(path)
        data = input_data.rename({
            "scan_time": "time",
        })[["time", "reflectivity", "path_attenuation", "freezing_level"]]

        # Need to expand scan time to full dimensions.
        time, _ = xr.broadcast(data.time, data.longitude)
        data["time"] = time

        refl = data.reflectivity.data
        refl = gaussian_filter1d(data.reflectivity.data, 5, -2)
        refl[refl < 0] = np.nan
        data["reflectivity"].data[:] = refl
        data = data[{"bins": slice(0, None, 6)}]

        data = resample_and_split(
            data,
            domain[self.scale],
            time_step,
            radius_of_influence=self.radius_of_influence,
            include_swath_center_coords=True
        )
        if data is None:
            return None

        for time_ind  in range(data.time.size):

            data_t = data[{"time": time_ind}]

            refl = data_t.reflectivity.data
            if np.isfinite(refl).any(-2).sum() < 100:
                LOGGER.info(
                    "Less than 100 valid pixels in training sample @ %s.",
                    data_t.time.data
                )
                continue

            encoding = {
                "reflectivity": {"dtype": "int8", "zlib": True, "_FillValue": 255},
                "path_attenuation": {"dtype": np.float32, "zlib": True},
                "freezing_level": {"dtype": np.float32, "zlib": True},
                "col_inds_swath_center": {"dtype": "int16"},
                "row_inds_swath_center": {"dtype": "int16"},
            }
            filename = get_output_filename(
                self.name, data.time[time_ind].data, time_step
            )

            LOGGER.info(
                "Writing training samples to %s.",
                output_folder / filename
            )
            data_t.to_netcdf(output_folder / filename, encoding=encoding)


DPR = GPMDPR(bands=[0, 1])
