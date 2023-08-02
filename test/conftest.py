
import pytest
import numpy as np
from scipy.fft import idctn
import xarray as xr

from cimr import areas
from cimr.data import mrms
from cimr.data import cpcir


def random_spectral_field(
        size,
        min_var
):
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
    wn = np.sqrt(wny[..., None] ** 2 + wnx ** 2)
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
        np.timedelta64(30, "m")
    )
    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:

        rqi = random_spectral_field((lats.size, lons.size), 10)
        rqi = rqi - rqi.min() / (rqi.max() - rqi.min())

        sp = random_spectral_field((lats.size, lons.size), 10)

        filename = mrms.get_output_filename(time)
        dataset = xr.Dataset({
            "latitude": (("latitude"), lats),
            "longitude": (("longitude"), lons),
            "surface_precip": (("latitude", "longitude"), sp),
            "rqi": (("latitude", "longitude"), rqi),
            "time": ((), time)
        })
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
        np.timedelta64(30, "m")
    )

    lons, lats = areas.CONUS_4.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:

        tbs = random_spectral_field(
            (lats.size, lons.size),
            10
        )[None]
        filename = cpcir.get_output_filename(time.item(), 30)
        dataset = xr.Dataset({
            "time": ((), time),
            "tbs": (("channels", "y", "x"), tbs),
        })
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
        np.timedelta64(120, "m")
    )

    lons, lats = areas.CONUS_8.get_lonlats()
    lons = lons[0]
    lats = lats[..., 0]

    for time in times:

        tbs = random_spectral_field(
            (lats.size, lons.size),
            10
        )[None]

        time_py = time.item()
        year = time_py.year
        month = time_py.month
        day = time_py.day
        hour = time_py.hour
        minute = time_py.minute
        filename = f"gmi_{year}{month:02}{day:02}_{hour:02}_{minute:02}.nc"
        dataset = xr.Dataset({
            "time": ((), time),
            "tbs": (("channels", "y", "x"), tbs),
        })
        dataset.to_netcdf(data_path / filename)


    return tmp_path
