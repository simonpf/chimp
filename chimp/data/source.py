"""
chimp.data.source
=================

Defines the base class for data sources. A data source is any
data product that can be downloaded and used to generate training
or validation data.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional, Union


import numpy as np
from pyresample import AreaDefinition
import xarray as xr


from pansat.geometry import Geometry
from pansat.utils import resample_data
from pansat.time import to_datetime64, to_timedelta64

from chimp.areas import Area
from chimp.data.utils import round_time
from chimp import extensions


LOGGER = logging.getLogger(__name__)

ALL_SOURCES = {}

class DataSource(ABC):
    """
    The data source base class keep track of all initiated source classes.
    """
    def __init__(self, name):
        self.name = name
        ALL_SOURCES[name] = self

    def find_files(
            self,
            start_time: np.datetime64,
            end_time: np.datetime64,
            time_step: np.timedelta64,
            roi: Optional[Geometry] = None,
            path: Optional[Path] = None
    ) -> List[Path]:
        """
        Find input data files from which to extract training data in a
        given time range.

        Args:
            start_time: Start time of the time range.
            end_time: End time of the time range.
            time_step: The time step of the retrieval.
            roi: An optional geometry object describing the region of interest
                that can be used to restriced the file selection a priori.
            path: If provided, files should be restricted to those available from
                the given path.

        Return:
            A list of locally available files from to extract CHIMP training
            data.
        """

    def process_file(
            self,
            path: Path,
            domain: Area,
            output_folder: Path,
            time_step: np.timedelta64
    ) -> None:
        """
        Extract training samples from a given source file.

        Args:
            path: A Path object pointing to the file to process.
            domain: An area object defining the training domain.
            output_folder: A path pointing to the folder to which to write
                the extracted training data.
            time_step: A timedelta object defining the retrieval time step.



        """

    def process_day(
        self,
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(minutes=15),
        include_scan_time=False,
    ):
        """
        Extract training data from a day of GOES observations.

        Args:
            domain: A domain object identifying the spatial domain for which
                to extract input data.
            year: The year
            month: The month
            day: The day
            output_folder: The folder to which to write the extracted
                observations.
            path: Not used, included for compatibility.
            time_step: The temporal resolution of the training data.
            include_scan_time: Not used.
        """
        start_time = datetime(year, month, day)
        end_time = start_time + timedelta(hours=23, minutes=59)
        start_time = to_datetime64(start_time)
        end_time = to_datetime64(end_time)
        time_step = to_timedelta64(time_step)

        input_files = self.find_files(
            start_time,
            end_time,
            time_step,
            roi=domain,
            path=path
        )
        failed = []
        for input_file in input_files:
            try:
                self.process_file(
                    input_file,
                    domain,
                    output_folder,
                    time_step
                )
            except Exception:
                LOGGER.exception(
                    "An error was encountered when processing file %s",
                    input_file
                )
                failed.append(input_file)
        return failed


    def find_training_files(self, path: Path) -> List[Path]:
        """
        Find training data files.

        Args:
            path: Path to the folder the training data for all input
                and reference datasets.

        Return:
            A list of found reference data files.
        """
        pattern = "*????????_??_??.nc"
        reference_files = sorted(
            list((path / self.name).glob(pattern))
        )
        return reference_files


def get_source(name: Union[str, DataSource]) -> DataSource:
    """
    Retrieve data source by name.

    Args:
        name: The name of a dataset for which to obtain the
            data source.

    Return:
        A DataSource object that can be used to extract data for a given
        dataset.
    """
    extensions.load()

    if isinstance(name, DataSource):
        return name
    if name in ALL_SOURCES:
        return ALL_SOURCES[name]

    raise ValueError(
        f"The data source '{name}' is currently not implemented. Available "
        f" sources are {list(ALL_SOURCES.keys())}."
    )
