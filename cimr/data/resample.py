"""
cimr.data.resample
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


def resample_tbs(domain, data, n_swaths=None, radius_of_influence=15e3):
    """
    Resample brightness temperatures (tbs) from swath to domain.

    Args:
        domain: pyresample.AreadDefinition defining the area to resample the
            swath to.
        data: An 'xarray.Dataset' containing the brightness temperatures,
            potentially, in multiple swaths.
        n_swaths: The number of swaths in the data.
        radius_of_influence: The radius of influence to use for the resampling.

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

    tbs = np.concatenate(swath_tbs, -1)
    dataset = xr.Dataset({
        "tbs": (("y", "x", "channels"), tbs),
        "swath_center_row_inds": (("swath_centers",), row_inds),
        "swath_center_col_inds": (("swath_centers",), col_inds),
    })
    return dataset


def resample_retrieval_targets(
        domain,
        data,
        targets=None,
        radius_of_influence=15e3):
    """
    Resample brightness temperatures (tbs) from swath to domain.

    Args:
        domain: pyresample.AreadDefinition defining the area to resample the
            swath to.
        data: An 'xarray.Dataset' containing the brightness temperatures,
            potentially, in multiple swaths.
        n_swaths: The number of swaths in the data.
        radius_of_influence: The radius of influence to use for the resampling.

    Return:
        An 'xarray.Dataset' containing the resampled brightness temperatures
        and the row and column indices of the swath centers.
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
