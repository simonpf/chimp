"""
chimp.plotting
==============

Defines

"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
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
