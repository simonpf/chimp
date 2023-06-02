"""
cimr.data.gmi
=============

This module implements the interface to extract GMI observations for
the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pansat.roi import find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l1c_gpm_gmi_r
from pyresample import geometry, kd_tree
import xarray as xr

from cimr.utils import round_time
from cimr.data.utils import make_microwave_band_array
from cimr.data.resample import resample_tbs

BANDS = {
    "mw_low": {
        0: ("tbs_s1", 1),  # 10.65 GHz, H
        1: ("tbs_s1", 0),  # 10.65 GHz, V
        2: ("tbs_s1", 3),  # 19 GHz, H
        3: ("tbs_s1", 2),  # 19 GHz, V
        4: ("tbs_s1", 4),  # 23 GHz, V
        5: ("tbs_s1", 6),  # 37 GHz, H
        6: ("tbs_s1", 5),  # 37 GHz, V
    },
    "mw_90": {
        0: ("tbs_s2", 1),  # 89 GHz, H
        1: ("tbs_s2", 0),  # 89 GHz, V
    },
    "mw_160": {
        0: ("tbs_s2", 3),  # 166 GHz, H
        1: ("tbs_s2", 2),  # 166 GHz, V
    },
    "mw_183": {
        2: ("tbs_s2", 4),  # 183 +/- 3
        4: ("tbs_s2", 5),  # 183 +/- 7
    },
}


def resample_swaths(domain, scene):
    """
    Resample GMI observations to 8 and 16 kilometer domains.

    Args:
        domain: A domain dict describing the domain for which to extract
            the training data.
        scene: An xarray.Dataset containing the observations over the desired
            domain.

    Return:
        A tuple ``tbs_r`` containing the ``tbs_s1`` observations
        excluding the 89 GHz channels resampled to the 16-km domain  and
        89-GHz observations and the ``tbs_s2`` observations resampled
        to the 8-km domain.
    """
    tbs_r = {}

    suffix = "_s1"
    lons = scene[f"longitude{suffix}"].data
    lats = scene[f"latitude{suffix}"].data
    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    tbs = scene[f"tbs{suffix}"].data
    tbs_r[f"tbs{suffix}"] = kd_tree.resample_nearest(
        swath, tbs[..., :-2], domain[16], radius_of_influence=16e3, fill_value=np.nan
    )

    suffix = "_s2"
    lons = scene[f"longitude{suffix}"].data
    lats = scene[f"latitude{suffix}"].data
    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    tbs = np.concatenate(
        [scene["tbs_s1"].data[..., -2:], scene["tbs_s2"].data[..., :]], axis=-1
    )
    tbs_r[f"tbs{suffix}"] = kd_tree.resample_nearest(
        swath, tbs, domain[8], radius_of_influence=16e3, fill_value=np.nan
    )
    return tbs_r


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

    tbs_r = tbs_r.transpose("channels", "y", "x", "swath_centers")

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


def process_file(domain, data, output_folder, time_step):
    """
    Extract training data from a single GMI L1C file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the L1C data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
        time_step: The time step between consecutive retrieval steps.
    """
    data = data.copy()
    data["latitude"] = data["latitude_s1"]
    data["longitude"] = data["longitude_s2"]

    scenes = find_overpasses(domain["roi"], data)

    for scene in scenes:
        tbs_r = resample_tbs(domain[8], data, n_swaths=2, radius_of_influence=15e3)
        time = scene.scan_time.mean().data.item()
        tbs_r.attrs = scene.attrs
        save_scene(time, tbs_r, output_folder, time_step)


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
    Extract GMI input observations for the CIMR retrieval.

    Args:
        domain: A domain dict specifying the area for which to
            extract GMI input data.
        year: The year.
        month: The month.
        day: The day.
        output_folder: The root of the directory tree to which to write
            the training data.
        path: Not used, included for compatibility.
        time_step: The time step between consecutive retrieval steps.
    """
    output_folder = Path(output_folder) / "gmi"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    provider = GesdiscProvider(l1c_gpm_gmi_r)
    product_files = provider.get_files_in_range(start_time, end_time)
    # For all file on given day.
    for filename in product_files:
        # Check if swath covers ROI.
        swath = parse_swath(provider.download_metadata(filename))
        if swath.intersects(domain["roi_poly"].to_geometry()):
            # Extract observations
            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                provider.download_file(filename, tmp / filename)
                data = l1c_gpm_gmi_r.open(tmp / filename)
                process_file(domain, data, output_folder, time_step)
