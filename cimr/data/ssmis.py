"""
cimr.data.ssmis
===============

This module implements the interface to extract SSMIS observations for
the CIMR training data generation.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from pansat.roi import find_overpasses
from pansat.metadata import parse_swath
from pansat.download.providers import GesdiscProvider
from pansat.products.satellite.gpm import l1c_f17_ssmis, l1c_f18_ssmis
from pyresample import geometry, kd_tree
import xarray as xr

from cimr.utils import round_time
from cimr.definitions import N_CHANS


# The pansat products providing access to SSMIS L1C data.
SSMIS_PRODUCTS = [
    l1c_f17_ssmis,
    l1c_f18_ssmis,
]

# Maps CIMR low-frequency band-indices  to the variables and channel
# indices of the SSMIS L1C data.
BANDS_LOW = {}

# Maps CIMR hi-frequency band-indices to the variables and channel
# indices of the SSMIS L1C data.
BANDS = {
    "mw_low": {
        2: ("tbs_s1", 1),  # 19 H
        3: ("tbs_s1", 0),  # 19 V
        4: ("tbs_s1", 2),  # 22.2 V
        5: ("tbs_s2", 1),  # 37 V
        6: ("tbs_s2", 0),  # 37 H
    },
    "mw_90": {
        0: ("tbs_s4", 1),  # 90 H
        1: ("tbs_s4", 0),  # 90 V
    },
    "mw_160": {
        0: ("tbs_s3", 0),  # 150 GHz
    },
    "mw_183": {0: ("tbs_s3", 1), 2: ("tbs_s3", 2), 4: ("tbs_s3", 3)},
}


def make_band_array(domain, band):
    """
    Create an empty array to hold  microwave observation data over the
    given domain.

    Args:
        domain: A domain dict describing the domain for which training data
            is created.
        band: The band name, i.e. one of ['mw_low', 'mw_90', 'mw_160', 'mw_183']

    Return:
        An array filled with nan's with the shape of the observation data
        for the given domain.
    """
    n_chans = N_CHANS[band]
    if band == "mw_low":
        shape = domain[16].shape
    else:
        shape = domain[8].shape
    shape = shape + (n_chans,)

    return xr.DataArray(np.nan * np.ones(shape, dtype=np.float32))


def resample_swaths(domain, scene):
    """
    Resample SSMIS swaths to 8 and 16 kilometer domains.

    Args:
        domain: A domain dict describing the domain for which to extract
            the training data.
        scene: An xarray.Dataset containing the observations over the desired
            domain.

    Return:
        A tuple ``tbs_r`` containing the ``tbs_s1, tbs_s2`` observations
        resampled to the 16-km domain  and ``tbs_s3, tbs_s4`` resampled
        to the 8-km domain.
    """
    tbs_r = {}

    for swath_index in range(1, 3):
        suffix = f"_s{swath_index}"
        lons = scene[f"longitude{suffix}"].data
        lats = scene[f"latitude{suffix}"].data
        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        tbs = scene[f"tbs{suffix}"].data
        tbs_r[f"tbs{suffix}"] = kd_tree.resample_nearest(
            swath, tbs, domain[16], radius_of_influence=16e3, fill_value=np.nan
        )

    for swath_index in range(3, 5):
        suffix = f"_s{swath_index}"
        lons = scene[f"longitude{suffix}"].data
        lats = scene[f"latitude{suffix}"].data
        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        tbs = scene[f"tbs{suffix}"].data
        tbs_r[f"tbs{suffix}"] = kd_tree.resample_nearest(
            swath, tbs, domain[8], radius_of_influence=32e3, fill_value=np.nan
        )
    return tbs_r


def save_scene(domain, scene, tbs_r, output_folder):
    """
    Save training data scene.

    Args:
        scene: xarray.Dataset containing the overpass scene over the
            domain. This data is only used  to extract the meta data of the
            training scene.
        tbs_low: A dict containing the S1 and S2 swaths resampled to
            the 8 km grid.
        tbs_high: A dict containing the S3 and S4 swaths resampled
            to the 16 km grid.
        output_folder: The folder to which to write the training data.
    """
    # Only use scenes with substantial information
    if np.any(np.isfinite(tbs_r["tbs_s3"][..., 0]), axis=-1).sum() < 100:
        return None

    # Extract observations and map to CIMR bands.
    results = {}
    for band, chans in BANDS.items():
        obs = make_band_array(domain, band)
        for ind_out, (name, ind_in) in chans.items():
            obs[..., ind_out] = tbs_r[name][..., ind_in]
        results[band] = obs

    dataset = xr.Dataset()
    for band in BANDS:
        if band == "mw_low":
            dims = ("y_16", "x_16", f"channels_{band}")
        else:
            dims = ("y_8", "x_8", f"channels_{band}")
        dataset[band] = (dims, results[band].data)

    time = scene.scan_time.mean().item()
    time_15 = round_time(time)
    year = time_15.year
    month = time_15.month
    day = time_15.day
    hour = time_15.hour
    minute = time_15.minute

    filename = f"mw_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
    output_filename = Path(output_folder) / filename

    if output_filename.exists():
        dataset_out = xr.load_dataset(output_filename)
        for band in BANDS.keys():
            if band in dataset_out:
                var_out = dataset_out[band]
                var_in = dataset[band]
                missing = ~np.isfinite(var_in.data)
                var_out.data[~missing] = var_in.data[~missing]
            else:
                dataset_out[band] = dataset[band]
        dataset_out.attrs["sensor"] += ", " + scene.attrs["InstrumentName"]
        dataset_out.attrs["satellite"] += ", " + scene.attrs["SatelliteName"]
        dataset_out.to_netcdf(output_filename)
    else:
        comp = {
            "dtype": "int16",
            "scale_factor": 0.01,
            "zlib": True,
            "_FillValue": -99,
        }
        dataset.attrs["sensor"] = scene.attrs["InstrumentName"]
        dataset.attrs["satellite"] = scene.attrs["SatelliteName"]
        encoding = {band: comp for band in BANDS}
        dataset.to_netcdf(output_filename, encoding=encoding)

    return None


def process_file(domain, data, output_folder):
    """
    Extract training data from a single SSMIS L1C file.

    Args:
        domain: A domain dict describing the area for which to extract
            the training data.
        data: An 'xarray.Dataset' containing the L1C data.
        output_folder: Path to the root of the directory tree to which
            to write the training data.
    """
    data = data.copy()
    data["latitude"] = data["latitude_s1"]
    data["longitude"] = data["longitude_s2"]

    scenes = find_overpasses(domain["roi"], data)

    for scene in scenes:
        tbs_r = resample_swaths(domain, data)
        save_scene(domain, scene, tbs_r, output_folder)


def process_day(domain, year, month, day, output_folder, path=None):
    """
    Extract SSMIS input observations for training the CIMR retrieval.

    Args:
        domain: A domain dict specifying the area for which to
            extract SSMIS input data.
        year: The year.
        month: The month>
        day: The day.
        output_folder: The root of the directory tree to which to write
            the training data.
        path: Not used, included for compatibility.
    """
    output_folder = Path(output_folder) / "microwave"
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    start_time = datetime(year, month, day)
    end_time = datetime(year, month, day) + timedelta(hours=23, minutes=59)
    # Iterate over platform.
    for product in SSMIS_PRODUCTS:
        provider = GesdiscProvider(product)
        product_files = provider.get_files_in_range(start_time, end_time)
        # For all file on given day.
        for filename in product_files:
            # Check if swath covers ROI.
            swath = parse_swath(provider.download_metadata(filename))
            if swath.intersects(domain["roi_poly"].to_geometry()):
                # Extract observations
                with TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    print("processing: ", filename)
                    provider.download_file(filename, tmp / filename)
                    data = product.open(tmp / filename)
                    process_file(domain, data, output_folder)
