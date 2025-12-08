"""
chimp.data.dem
==============

The 'dem' input dataset provides an 'elevation' input containing an elevation field from
a digital elevation model.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pansat.time import TimeRange
from pansat.geometry import Geometry
from pansat.products import dem
from pansat.utils import resample_data
import xarray as xr


from chimp.areas import Area
from chimp.data.input import StaticInputDataset


class NOAAGlobe(StaticInputDataset):
    """
    Interface for downloading and loading the NOAA GLOBE digital elevation model.
    """
    def __init__(self):
        """
        Args:
            name: The name of the reference dataset.
            scale: The spatial scale of the data.
            targets: A list of the retrieval targets provided by the dataset.
            quality_index: Name of the field hodling the quality index.
        """
        StaticInputDataset.__init__(self, "dem", "elevation", 4, ["elevation"], n_dim=2)
        self.n_channels = 1


    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Area] = None,
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
        time_range = TimeRange("2020-01-01")
        recs = dem.globe.find_files(time_range=time_range, roi=roi.roi)
        return [recs]


    def process_file(
        self,
        path: Path,
        domain: Area,
        output_folder: Path,
        time_step: np.timedelta64
    ):
        if not isinstance(path, list):
            recs = [path]
        else:
            recs = path
        recs = [rec.get() for rec in recs]
        data = xr.merge([dem.globe.open(rec) for rec in recs])
        data_r = resample_data(data, domain[self.scale], new_dims=("y", "x"))

        output_folder = Path(output_folder) / self.name
        output_folder.mkdir(exist_ok=True)
        file_path = output_folder / "elevation.nc"
        data_r.to_netcdf(file_path, encoding={"elevation": {"dtype": "float32", "zlib": True}})


    def find_training_files(
            self,
            path: Path,
            times: Optional[np.ndarray] = None
    )  -> Tuple[np.ndarray, List[Path]]:
        """
        Find DEM elevation files.

        Args:
            path: Path to the folder containing the training data.
            times: The available reference data times.

        Return:
            A tuple ``(times, paths)`` containing the times for which training
            files are available and the paths pointing to the corresponding file.
        """
        paths = [path / "dem" / "elevation.nc" for _ in times]
        return times, paths


globe = NOAAGlobe()
