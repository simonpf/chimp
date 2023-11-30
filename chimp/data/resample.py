"""
chimp.data.resample
==================

Functions to resample swath data to specific domains.
"""
import numpy as np
from pyresample import geometry
from pyresample import kd_tree
import xarray as xr


def resample_swath_center(domain, lons, lats, radius_of_influence=15e3):
    """
    Resample center of swath to domain.

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


def resample_tbs(
        domain,
        data,
        n_swaths=None,
        radius_of_influence=15e3,
        include_scan_time=True
):
    """
    Resample brightness temperatures (tbs) from swath to domain.

    Args:
        domain: pyresample.AreadDefinition defining the area to resample the
            swath to.
        data: An 'xarray.Dataset' containing the brightness temperatures,
            potentially, in multiple swaths.
        n_swaths: The number of swaths in the data.
        radius_of_influence: The radius of influence to use for the resampling.
        include_scan_time: Flag indicating whether or not to include the
            scan time in the results.

    Return:
        An 'xarray.Dataset' containing the resampled brightness temperatures
        and the row and column indices of the swath centers.
    """
    if n_swaths is None:
        n_swaths = 1
        no_swaths = True
    else:
        no_swaths = False

    swath_tbs = []
    scan_time = None

    for swath_ind in range(1, n_swaths + 1):

        if no_swaths:
            suffix = f""
            lons = data[f"longitude"].data
            lats = data[f"latitude"].data
            tbs = data[f"tbs"].data
        else:
            suffix = f"_s{swath_ind}"
            lons = data[f"longitude{suffix}"].data
            lats = data[f"latitude{suffix}"].data
            tbs = data[f"tbs{suffix}"].data

        if swath_ind == 1:
            row_inds, col_inds = resample_swath_center(
                domain,
                lons,
                lats,
                radius_of_influence=radius_of_influence
            )

        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        tbs = kd_tree.resample_nearest(
            swath,
            tbs,
            domain,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        swath_tbs.append(tbs)

        if swath_ind == 1 and include_scan_time:
            scan_time = data["scan_time"].data[..., None]
            scan_time = np.broadcast_to(scan_time, lons.shape)
            scan_time = kd_tree.resample_nearest(
                swath,
                scan_time,
                domain,
                radius_of_influence=radius_of_influence,
                fill_value=np.datetime64("NAT")
            )


    tbs = np.concatenate(swath_tbs, -1)
    dataset = xr.Dataset({
        "tbs": (("y", "x", "channels"), tbs),
        "swath_center_row_inds": (("swath_centers",), row_inds),
        "swath_center_col_inds": (("swath_centers",), col_inds),
    })

    if scan_time is not None:
        dataset["scan_time"] = (("y", "x"), scan_time)

    return dataset


def resample_retrieval_targets(
        domain,
        data,
        targets=None,
        radius_of_influence=15e3):
    """
    Resample retrieval targets.

    Args:
        domain: pyresample.AreadDefinition defining the grid to resample the
            target data to.
        data: An 'xarray.Dataset' containing the retrieval targets.
        n_swaths: The number of swaths in the data.
        radius_of_influence: The radius of influence to use for the resampling.

    Return:
        An 'xarray.Dataset' containing the resampled target variables.
    """
    if targets is None:
        targets = ["surface_precip"]
    lons = data[f"longitude"].data
    lats = data[f"latitude"].data

    results = {}

    row_inds, col_inds = resample_swath_center(
        domain,
        lons,
        lats,
        radius_of_influence=radius_of_influence
    )

    for target in targets:
        swath = geometry.SwathDefinition(lons=lons, lats=lats)
        data_r = kd_tree.resample_nearest(
            swath,
            data[target].data,
            domain,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        results[target] = (("y", "x"), data_r)

    results["swath_center_row_inds"] = (("swath_centers",), row_inds)
    results["swath_center_col_inds"] = (("swath_centers",), col_inds)

    return xr.Dataset(results)


def resample_data(dataset, target_grid, radius_of_influence=5e3):
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
    lons_t, lats_t = target_grid.get_lonlats()

    valid_pixels = (
        (lons_t >= lons.min())
        * (lons_t <= lons.max())
        * (lats_t >= lats.min())
        * (lats_t <= lats.max())
    )

    swath = geometry.SwathDefinition(lons=lons, lats=lats)
    target = geometry.SwathDefinition(
        lons=lons_t[valid_pixels],
        lats=lats_t[valid_pixels]
    )

    info = kd_tree.get_neighbour_info(
        swath, target, radius_of_influence=radius_of_influence, neighbours=1
    )
    ind_in, ind_out, inds, _ = info

    dims = ("latitude", "longitude")
    resampled = {}
    resampled["latitude"] = (("latitude",), lats_t[:, 0])
    resampled["longitude"] = (("longitude",), lons_t[0, :])

    for var in dataset:
        data = dataset[var].data
        if data.ndim == 1:
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

        data_full = np.zeros(target_grid.shape + data.shape[2:], dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            data_full = np.nan * data_full
        elif np.issubdtype(dtype, np.datetime64):
            data_full[:] = np.datetime64("NaT")
        elif dtype == np.int8:
            data_full[:] = -1
        else:
            data_full[:] = -9999

        data_full[valid_pixels] = data_r
        resampled[var] = (dims + dataset[var].dims[2:], data_full)

    return xr.Dataset(resampled)
