"""
chimp.plotting
==============

Defines

"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


def set_style():
    """
    Set the CHIMP matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "chimp.mplstyle")

def add_ticks(
        ax: plt.Axes,
        lons: List[float],
        lats: list[float],
        left=True,
        bottom=True
) -> None:
    import cartopy.crs as ccrs
    """
    Add tick to cartopy Axes object.

    Args:
        ax: The Axes object to which to add the ticks.
        lons: The longitude coordinate at which to add ticks.
        lats: The latitude coordinate at which to add ticks.
        left: Whether or not to draw ticks on the y-axis.
        bottom: Whether or not to draw ticks on the x-axis.
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='none')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left
    gl.bottom_labels = bottom
    gl.xlocator = FixedLocator(lons)
    gl.ylocator = FixedLocator(lats)

cmap_precip = sns.cubehelix_palette(start=1.50, rot=-0.9, as_cmap=True, hue=0.8, dark=0.2, light=0.9)
cmap_tbs = sns.cubehelix_palette(start=2.2, rot=0.9, as_cmap=True, hue=1.3, dark=0.2, light=0.8, reverse=True)
cmap_tbs = sns.color_palette("rocket", as_cmap=True)


def scale_bar(
        ax,
        length,
        location=(0.5, 0.05),
        linewidth=3,
        height=0.01,
        border=0.05,
        border_color="k",
        parts=4,
        zorder=50,
        textcolor="k"
):
    """
    Draw a scale bar on a cartopy map.

    Args:
        ax: The matplotlib.Axes object to draw the axes on.
        length: The length of the scale bar in meters.
        location: A tuple ``(h, w)`` defining the fractional horizontal
            position ``h`` and vertical position ``h`` in the given axes
            object.
        linewidth: The width of the line.
    """
    import cartopy.crs as ccrs
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    lon_c = lon_min + (lon_max - lon_min) * location[0]
    lat_c = lat_min + (lat_max - lat_min) * location[1]
    transverse_merc = ccrs.TransverseMercator(lon_c, lat_c)

    x_min, x_max, y_min, y_max = ax.get_extent(transverse_merc)

    x_c = x_min + (x_max - x_min) * location[0]
    y_c = y_min + (y_max - y_min) * location[1]

    x_left = x_c - length / 2
    x_right = x_c  + length / 2

    def to_axes_coords(point):
        crs = ax.projection
        p_data = crs.transform_point(*point, src_crs=transverse_merc)
        return ax.transAxes.inverted().transform(ax.transData.transform(p_data))

    def axes_to_lonlat(point):
        p_src = ax.transData.inverted().transform(ax.transAxes.transform(point))
        return ccrs.PlateCarree().transform_point(*p_src, src_crs=ax.projection)


    left_ax = to_axes_coords([x_left, y_c])
    right_ax = to_axes_coords([x_right, y_c])

    l_ax = right_ax[0] - left_ax[0]
    l_part = l_ax / parts



    left_bg = [
        left_ax[0] - border,
        left_ax[1] - height / 2 - border
    ]

    background = Rectangle(
        left_bg,
        l_ax + 2 * border,
        height + 2 * border,
        facecolor="none",
        transform=ax.transAxes,
        zorder=zorder
    )
    ax.add_patch(background)

    for i in range(parts):
        left = left_ax[0] + i * l_part
        bottom = left_ax[1] - height / 2

        color = "k" if i % 2 == 0 else "w"
        rect = Rectangle(
            (left, bottom),
            l_part,
            height,
            facecolor=color,
            edgecolor=border_color,
            transform=ax.transAxes,
            zorder=zorder
        )
        ax.add_patch(rect)

    x_bar = [x_c - length / 2, x_c + length / 2]
    x_text = 0.5 * (left_ax[0] + right_ax[0])
    y_text = left_ax[1] + 0.1 * height + 1 * border
    ax.text(x_text,
            y_text,
            f"{length / 1e3:g} km",
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            color=textcolor
    )
