import os

import pytest
import numpy as np
from scipy.fft import idctn
import xarray as xr

from chimp import areas
from chimp.data import mrms
from chimp.data import cpcir

from chimp.data import get_input_dataset, get_reference_dataset
from chimp.data.training_data import SingleStepDataset, SequenceDataset
from chimp.data.utils import get_output_filename


HAS_PANSAT_PASSWORD = "PANSAT_PASSWORD" in os.environ
NEEDS_PANSAT_PASSWORD = pytest.mark.skipif(
    not HAS_PANSAT_PASSWORD, reason="PANSAT_PASSWORD not set."
)


def random_spectral_field(size, min_var):
    """
    Create random spectral fields with minimum scale of spatial
    variation.

    Args:
        size: Tuple ``(h, w)`` specifying the height and widt of the field.
        min_var: Minimum scale of spatial variability

    Return:
        A random fields ``field``, which exhibits spatial variability
        across all scales larger than 'min_var'.
    """
    wny = 0.5 * np.arange(size[0]) / size[0]
    wnx = 0.5 * np.arange(size[1]) / size[1]
    wn = np.sqrt(wny[..., None] ** 2 + wnx**2)
    scale = 1 / wn

    coeffs = np.random.rand(*size)
    coeffs *= np.sqrt(scale)
    coeffs[:, 0] = 0.0
    coeffs[0, :] = 0.0
    coeffs[scale < min_var] = 0.0

    field = idctn(coeffs, norm="ortho")

    return field


@pytest.fixture
def mrms_surface_precip_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MRMS reference data.
    """
    data_path = tmp_path / "mrms"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-01T12:00:00", "s"),
        np.timedelta64(30, "m"),
    )
    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        rqi = random_spectral_field((lats.size, lons.size), 10)
        rqi += rqi.min()
        med = np.median(rqi)
        rqi = np.minimum(rqi, med) / med

        sp = random_spectral_field((lats.size, lons.size), 10).astype("float32")

        filename = get_output_filename("mrms", time, np.timedelta64(30, "m"))
        dataset = xr.Dataset(
            {
                "latitude": (("latitude"), lats),
                "longitude": (("longitude"), lons),
                "surface_precip": (("latitude", "longitude"), sp),
                "rqi": (("latitude", "longitude"), rqi),
                "time": ((), time.astype("datetime64[ns]")),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def mrms_reflectivity_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MRMS reflectivity data.
    """
    data_path = tmp_path / "mrms"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-01T12:00:00", "s"),
        np.timedelta64(30, "m"),
    )
    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        rqi = random_spectral_field((lats.size, lons.size), 10)
        rqi += rqi.min()
        med = np.median(rqi)
        rqi = np.minimum(rqi, med) / med

        refl = random_spectral_field((lats.size, lons.size), 10).astype("float32")

        filename = get_output_filename("mrms", time, np.timedelta64(30, "m"))
        dataset = xr.Dataset(
            {
                "latitude": (("latitude"), lats),
                "longitude": (("longitude"), lons),
                "reflectivity": (("latitude", "longitude"), refl),
                "rqi": (("latitude", "longitude"), rqi),
                "time": ((), time.astype("datetime64[ns]")),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def cpcir_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of CPCIR input data.
    """
    data_path = tmp_path / "cpcir"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(30, "m"),
    )

    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = random_spectral_field((lats.size, lons.size), 10)[..., None].astype(
            "float32"
        )
        filename = get_output_filename("cpcir", time.item(), np.timedelta64(30, "m"))
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("y", "x", "channels"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def gmi_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of GMI input data.
    """
    data_path = tmp_path / "gmi"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(120, "m"),
    )

    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(13)],
            axis=-1
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"gmi_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("y", "x", "channels"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path

@pytest.fixture
def cmb_surface_precip_data(tmp_path):
    """
    Initialize a temporary directory with random reference data files
    of CMB input data.
    """
    data_path = tmp_path / "cmb"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-02T00:00:00", "s"),
        np.timedelta64(90, "m"),
    )

    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        sp = random_spectral_field((lats.size, lons.size), 10).astype(
            "float32"
        )
        filename = get_output_filename("cmb", time.item(), np.timedelta64(30, "m"))
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "surface_precip": (("y", "x"), sp),
                "row_inds_swath_center": (
                    ("center_indices",),
                    np.linspace(0, lats.size - 1, 128).astype(np.int64)
                ),
                "col_inds_swath_center": (
                    ("center_indices",),
                    np.linspace(0, lons.size - 1, 128).astype(np.int64)
                )
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def mhs_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MHS input data.
    """
    data_path = tmp_path / "mhs"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:00:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(120, "m"),
    )

    lons, lats = areas.CONUS_8.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(5)]
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"mhs_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("channels", "y", "x"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def ssmis_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MHS input data.
    """
    data_path = tmp_path / "ssmis"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:30:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(120, "m"),
    )

    lons, lats = areas.CONUS_8.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(11)]
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"ssmis_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("channels", "y", "x"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def amsr2_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MHS input data.
    """
    data_path = tmp_path / "amsr2"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:30:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(60, "m"),
    )

    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(12)]
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"amsr2_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("channels", "y", "x"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def atms_data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MHS input data.
    """
    data_path = tmp_path / "atms"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:30:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(60, "m"),
    )

    lons, lats = areas.CONUS_16.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(9)]
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"atms_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("channels", "y", "x"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def _data(tmp_path):
    """
    Initialize a temporary directory with random training data files
    of MHS input data.
    """
    data_path = tmp_path / "amsr2"
    data_path.mkdir()

    times = np.arange(
        np.datetime64("2020-01-01T00:30:00", "s"),
        np.datetime64("2020-01-01T06:00:00", "s"),
        np.timedelta64(60, "m"),
    )

    lons, lats = areas.CONUS_8.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:
        tbs = np.stack(
            [random_spectral_field((lats.size, lons.size), 10) for _ in range(11)]
        )

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"amsr2_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset(
            {
                "time": ((), time.astype("datetime64[ns]")),
                "tbs": (("channels", "y", "x"), tbs),
            }
        )
        dataset.to_netcdf(data_path / filename)

    return tmp_path


@pytest.fixture
def training_data_multi(
    cpcir_data,
    gmi_data,
    mhs_data,
    ssmis_data,
    amsr2_data,
    atms_data,
    mrms_surface_precip_data,
):
    """
    Fixture providing a sequence dataset with inputs from CPCIR, GMI and
    MRMS outputs.
    """
    dataset = SingleStepDataset(
        mhs_data,
        inputs=["mhs", "cpcir", "gmi", "ssmis", "atms", "amsr2"],
        reference_data="mrms",
        window_size=256,
        missing_value_policy="missing",
    )
    return dataset


@pytest.fixture
def training_data_seq(gmi_data, cpcir_data, mrms_surface_precip_data):
    """
    Fixture providing a sequence dataset with inputs from CPCIR, GMI and
    MRMS outputs.
    """
    dataset = SequenceDataset(
        gmi_data,
        inputs=["gmi", "cpcir"],
        reference_data="mrms",
        window_size=128,
        missing_value_policy="masked",
        sequence_length=4,
        forecast=0,
    )
    return dataset
