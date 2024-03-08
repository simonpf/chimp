"""
chimp.data.resample
==================

Functions to resample swath data to specific domains.
"""
import numpy as np
from pyresample import geometry, kd_tree, AreaDefinition, SwathDefinition
import xarray as xr


from .utils import round_time


def resample_swath_centers(domain, lons, lats, radius_of_influence=15e3):
    """
    Resample centers of swath to domain.

    Args:
        domain: pyresample.AreadDefinition defining the area to resample the
            swath to.
        lons: 2D array containing the longitude coordinates of the swath.
        lats: 2D array containing the latitude coordinates of the swath.
        radius_of_influence: Radius of influence to use for resampling.

    Return:
        A tuple ``(row_indices, col_indices)`` containing the row and column
        indices of the swath center.
    """
    n_pixels = lons.shape[-1]
    lons_c = lons[:, n_pixels // 2]
    lats_c = lats[:, n_pixels // 2]
    swath = geometry.SwathDefinition(lons=lons_c, lats=lats_c)

    _, _, indices, dists = kd_tree.get_neighbour_info(
        domain,
        swath,
        radius_of_influence=radius_of_influence,
        neighbours=1
    )

    n_cols = domain.shape[1]
    col_indices = indices % n_cols
    row_indices = indices // n_cols
    return row_indices, col_indices


def resample_data(
    dataset, target_grid, radius_of_influence=5e3, new_dims=("latitude", "longitude")
) -> xr.Dataset:
    """
    Resample xarray.Dataset data to global grid.

    Args:
        dataset: xr.Dataset containing data to resample to global grid.
        target_grid: A pyresample.AreaDefinition defining the global grid
            to which to resample the data.

    Return:
        An xarray.Dataset containing the give dataset resampled to
        the global grid.
    """
    lons = dataset.longitude.data
    lats = dataset.latitude.data

    if "latitude" in dataset.dims:
        dataset = dataset.transpose(..., "latitude", "longitude")
        lons, lats = np.meshgrid(lons, lats)

    if isinstance(target_grid, tuple):
        lons_t, lats_t = target_grid
        shape = lons_t.shape
    else:
        lons_t, lats_t = target_grid.get_lonlats()
        shape = target_grid.shape

    valid_pixels = (
        (lons_t >= np.nanmin(lons))
        * (lons_t <= np.nanmax(lons))
        * (lats_t >= np.nanmin(lats))
        * (lats_t <= np.nanmax(lats))
    )

    swath = SwathDefinition(lons=lons, lats=lats)
    target = SwathDefinition(lons=lons_t[valid_pixels], lats=lats_t[valid_pixels])

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    resampled = {}
    resampled["latitude"] = (("latitude",), lats_t[:, 0])
    resampled["longitude"] = (("longitude",), lons_t[0, :])

    for var in dataset:
        if var in ["latitude", "longitude"]:
            continue
        data = dataset[var].data
        if data.ndim == 1 and lons.ndim > 1:
            data = np.broadcast_to(data[:, None], lons.shape)

        dtype = data.dtype
        if np.issubdtype(dtype, np.datetime64):
            fill_value = np.datetime64("NaT")
        elif np.issubdtype(dtype, np.integer):
            fill_value = -9999
        elif dtype == np.int8:
            fill_value = -1
        else:
            fill_value = np.nan

        data_r = kd_tree.get_sample_from_neighbour_info(
            "nn", target.shape, data, ind_in, ind_out, inds, fill_value=fill_value
        )

        data_full = np.zeros(shape + data.shape[lons.ndim :], dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            data_full = np.nan * data_full
        elif np.issubdtype(dtype, np.datetime64):
            data_full[:] = np.datetime64("NaT")
        elif dtype == np.int8:
            data_full[:] = -1
        else:
            data_full[:] = -9999

        data_full[valid_pixels] = data_r
        resampled[var] = (new_dims + dataset[var].dims[lons.ndim :], data_full)

    return xr.Dataset(resampled)


def resample_and_split(
        dataset: xr.Dataset,
        domain: AreaDefinition,
        time_step: np.timedelta64,
        radius_of_influence: float = 10e3,
        include_swath_center_coords: bool = False
) -> xr.Dataset:
    """
    Resample and discretize dataset in time.

    Args:
        dataset: A xarray.Dataset with a variables longitude and latitude defining
            the geolocations of all samples in the dataset and a variable time
            defining the corresponding measurement time.
        domain: A pyresample.AreaDefinition defining the grid to which to resample
            the data.
        time_step: A numpy.timedelta64 object defining the time step of the
            the retrieval
        radius_of_influence: The maximum allowed distance between input data
            samples and grid points to use in the resampling.
        include_swath_center_coords: Whether or not to include the coordinates
            of the swath center in the output.

    Return:
        An xarray dataset containing the data resampled to the given grid and
        discretized in time using the given time step.
    """
    if "longitude" in dataset.dims:
        dataset = dataset.transpose("latitude", "longitude", ...)
        lons = dataset.longitude.data
        lats = dataset.latitude.data
        lons, lats = xr.meshgrid(lons, lats)
    else:
        lons = dataset.longitude.data
        lats = dataset.latitude.data

    time = dataset.time.data
    start_time = round_time(time.min(), time_step)
    end_time = round_time(time.max(), time_step)

    # Spatial masking
    lons_t, lats_t = domain.get_lonlats()
    lon_min = lons_t.min()
    lon_max = lons_t.max()
    lat_min = lats_t.min()
    lat_max = lats_t.max()

    spatial_mask = (
        (lons >= lon_min) * (lons < lon_max) *
        (lats >= lat_min) * (lats < lat_max)
    )

    time = start_time
    results = []
    times = []

    while time <= end_time:

        if "time" in dataset.dims:
            data_t = dataset.interp(time=time.astype("datetime64[ns]"), method="nearest")
            mask = spatial_mask
            lons_swath = None
            lats_swath = None
        else:
            data_t = dataset.drop_vars("time")
            mask = (
                spatial_mask *
                (time - 0.5 * time_step <= dataset.time.data) *
                (time + 0.5 * time_step > dataset.time.data)
            )
            lons_swath = data_t.longitude.data[mask.any(-1)]
            lats_swath = data_t.latitude.data[mask.any(-1)]

        if mask.sum() == 0:
            time += time_step
            continue

        data_t = xr.Dataset({
            name: (("samples",) + da.dims[2:], da.data[mask])
            for name, da in data_t.variables.items() if da.ndim > 1
        })

        data_r = resample_data(
            data_t,
            domain,
            radius_of_influence=radius_of_influence,
            new_dims=("y", "x")
        )

        if include_swath_center_coords:
            if lons_swath is None:
                raise RuntimeError(
                    "Need swath data to resample swath centers."
                )
            row_inds, col_inds = resample_swath_centers(
                domain, lons_swath, lats_swath, radius_of_influence=radius_of_influence
            )
            inds = np.random.permutation(row_inds.size)[:128]
            if inds.size < 128:
                inds = np.random.choice(inds, size=128)
            row_inds = row_inds[inds].astype("int16")
            col_inds = col_inds[inds].astype("int16")
            data_r["row_inds_swath_center"] = (("center_indices",), row_inds)
            data_r["col_inds_swath_center"] = (("center_indices",), col_inds)

        results.append(data_r)
        times.append(time)
        time += time_step

    results = xr.concat(results, "time")
    results["time"] = (("time"), np.array(times).astype("datetime64[ns]"))
    return results
