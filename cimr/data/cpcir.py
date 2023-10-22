"""
cimr.data.cpcir
===============

This module implements functionality to extract IR brightness
temperature from the NCEP CPC merged IR dataset.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pansat.products.satellite.gpm import merged_ir
from pansat.time import to_datetime64, TimeRange
from pyresample import geometry, kd_tree, create_area_def
import xarray as xr

from cimr.utils import get_available_times, round_time
from cimr.data import Input, MinMaxNormalized


CPCIR_GRID = create_area_def(
    "cpcir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution= (0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="CPCIR grid",
)


def get_output_filename(time, round_minutes=15):
    """
    Get filename for training sample.

    Args:
        time: The observation time.
        round_minutes: The number of minutes to which to round
            the time in the filename.

    Return:
        A string specifying the filename of the training sample.
    """
    time_r = round_time(time, minutes=round_minutes)
    year = time_r.year
    month = time_r.month
    day = time_r.day
    hour = time_r.hour
    minute = time_r.minute

    filename = f"cpcir_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    return filename


def resample_data(domain, scene):
    """
    Resample CPCIR observations to 4 kilometer domain.

    Args:
        domain: A domain dict describing the domain for which to extract
            the training data.
        scene: An xarray.Dataset containing the observations over the desired
            domain.

    Return:
        An xarray dataset containing the resampled CPCIR Tbs.
    """
    tbs = scene.Tb.data
    tbs_r = kd_tree.resample_nearest(
        CPCIR_GRID,
        tbs[::-1],
        domain[4],
        radius_of_influence=5e3,
        fill_value=np.nan
    )
    return xr.Dataset({
        "time": ((), scene.time.data),
        "tbs": (("channels", "y", "x"), tbs_r[None]),
    })


def save_cpcir_data(data, output_folder, time_step):
    """
    Save CPCIR observation data to netcdf file.

    Args:
        data: A netcdf dataset containing resampled CPCIR
            brightness temperatures.
        output_folder: A Path object pointing to the directory
            in which to store the data.
        time: The
    """
    data.tbs.encoding = {
        "scale_factor": 150 / 254,
        "add_offset": 170,
        "zlib": True,
        "dtype": "uint8",
        "_FillValue": 255
    }
    filename = get_output_filename(data.time.data.item(), time_step)
    data.to_netcdf(output_folder / filename)


class CPCIRData(Input, MinMaxNormalized):
    """
    Represents input data derived from merged IR data.
    """
    def __init__(
            self,
            name: str,
            scale: int,
    ):
        MinMaxNormalized.__init__(self, name)
        Input.__init__(self, name, scale, "tbs", n_dim=2)
        self.scale = scale


    def process_day(
            self,
            domain: dict,
            year: int,
            month: int,
            day: int,
            output_folder: Path,
            path=None,
            time_step=timedelta(minutes=15),
            conditional=None,
            include_scan_time=False
    ):
        """
        Extract CPCIR input observations for the CIMR retrieval.

        Args:
            domain: A domain dict specifying the area for which to
                extract CPCIR input data.
            year: The year.
            month: The month.
            day: The day.
            output_folder: The root of the directory tree to which to write
                the training data.
            path: Not used, included for compatibility.
            time_step: The time step between consecutive retrieval steps.
            conditional: If provided, it should point to folder containing
                samples from another datasource. In this case, CPCIR input
                data will only be extracted for the times at which samples
                of the other dataset are available.
            include_scan_time: Ignored. Included for compatibility.
        """
        output_folder = Path(output_folder) / "cpcir"
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        existing_files = [
            f.name for f in output_folder.glob(f"cpcir_{year}{month:02}{day:02}*.nc")
        ]

        start_time = datetime(year, month, day)
        end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
        time_range = TimeRange(start_time, end_time)

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            if conditional is not None:
                available_times = get_available_times(conditional)
                for time in available_times:
                    start_time = time - timedelta(minutes=30)
                    end_time = time - timedelta(minutes=30)

                    files = mered_ir.fi(start_time, end_time)
                    local_files = []
                    for rec in files:
                        local_file = Path(tmp) / rec.filename
                        if not local_file.exists():
                            rec = rec.download(destination=tmp)
                        local_files.append(rec.local_path)

                    cpcir_data = xr.open_mfdataset(local_files)
                    cpcir_data = cpcir_data.interp(time=to_datetime64(time))
                    tbs_r = resample_data(domain, cpcir_data.compute())
                    save_cpcir_data(
                        cpcir_data,
                        output_folder,
                        time_step=time_step.minutes
                    )
            else:
                time = start_time
                while time < end_time:
                    output_filename = get_output_filename(to_datetime64(time))
                    if not (output_folder / output_filename).exists():
                        files = merged_ir.find_files(time_range)
                        local_paths = []
                        for rec in files:
                            local_path = tmp / rec.filename
                            if not local_path.exists():
                                rec = rec.download(destination=tmp)
                            local_paths.append(local_path)

                        cpcir_data = xr.open_mfdataset(
                            local_paths,
                        )
                        cpcir_data = cpcir_data.interp(time=to_datetime64(time))
                        tbs_r = resample_data(domain, cpcir_data.compute())
                        save_cpcir_data(
                            tbs_r,
                            output_folder,
                            time_step=time_step.seconds // 60
                        )
                    time = time + time_step

cpcir = CPCIRData("cpcir", 4)
