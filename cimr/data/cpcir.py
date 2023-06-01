"""
cimr.data.cpcir
===============
"""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pansat.products.satellite.gpm import gpm_mergeir
from pansat.download.providers import Disc2Provider
from pyresample import geometry, kd_tree, create_area_def
import xarray as xr

from cimr.utils import round_time


PROVIDER = Disc2Provider(gpm_mergeir)
CPCIR_GRID = create_area_def(
    "cpcir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution= (0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="CPCIR grid",
)


def get_output_filename(time):
    """
    Get filename for training sample.

    Args:
        time: The observation time.

    Return:
        A string specifying the filename of the training sample.
    """
    time_15 = round_time(time)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

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
        "tbs": (("y", "x"), tbs_r)
    })


def save_scene(time, tbs_r, output_folder, time_step):
    """
    Save training data scene.

    Args:
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        tbs_r: An xarray.Dataset containing the resampled brigthness
            temperatures.
        output_folder: The folder to which to write the training data.
    """
    minutes = time_step.seconds // 60
    time_15 = round_time(time, minutes=minutes)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"gmi_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    comp = {
        "dtype": "int16",
        "scale_factor": 0.01,
        "zlib": True,
        "_FillValue": -99,
    }
    encoding = {
        "tbs": comp,
        "swath_center_col_inds": {"dtype": "int16"},
        "swath_center_row_inds": {"dtype": "int16"}
    }
    tbs_r.to_netcdf(output_filename, encoding=encoding)
    return None


def save_data(data, output_folder, time):
    """
    """



def process_day(
        domain,
        year,
        month,
        day,
        output_folder,
        path=None,
        time_step=timedelta(minutes=15)
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
    """
    output_folder = Path(output_folder) / "cpcir"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    existing_files = [
        f.name for f in output_folder.glob(f"cpcir_{year}{month:02}{day:02}*.nc")
    ]

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)

    files = PROVIDER.get_files_in_range(start_time, end_time)
    with TemporaryDirectory() as tmp:
        for cpcir_file in files:
            local_file = Path(tmp) / cpcir_file
            PROVIDER.download_file(cpcir_file, local_file)
            cpcir_data = xr.load_dataset(local_file)
            for t_ind in range(2):

                cpcir_data_t = cpcir_data[{"time": t_ind}]
                time = cpcir_data_t.time.data.item()

                tbs_r = resample_data(domain, cpcir_data_t)
                tbs_r.tbs.data[:] = np.clip(tbs_r.tbs.data, 170, 320)
                tbs_r.tbs.encoding = {
                    "scale_factor": 150 / 254,
                    "add_offset": 170,
                    "zlib": True,
                    "dtype": "uint8",
                    "_FillValue": 255
                }
                filename = get_output_filename(time)
                tbs_r.to_netcdf(output_folder / filename)
