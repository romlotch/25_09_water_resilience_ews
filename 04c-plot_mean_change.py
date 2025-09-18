import re 
import os
import glob
import scipy
import pickle
import argparse
import rasterio 
import rioxarray
import regionmask
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sn
import geopandas as gpd
from datetime import datetime
from itertools import product

import statsmodels.api as sm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import colorsys
from skimage.color import rgb2lab, lab2rgb

from matplotlib.colors import to_rgb, to_hex
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.interpolate import make_interp_spline
from skimage.color import rgb2lab, lab2rgb

import warnings 

""" 
Plots and saves mean change of the EWS. 

Inputs: 
    dataset_path: EWS file path to plot
    variable: Variable (e.g. sm, Et, precip)
    output_path: Directory to save figures to. 

E.g.
    python 04c-plot_mean_change.py --dataset_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_meanchange.zarr' --variable 'sm' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2/meanchange'
    python 04c-plot_mean_change.py --dataset_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_meanchange.zarr' --variable 'Et' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2/meanchange'
    python 04c-plot_mean_change.py --dataset_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_meanchange.zarr' --variable 'precip' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2/meanchange'
"""


INDICATOR_STATS = ("ac1", "std", "skew", "kurt", "fd")
LABELS = {"ac1": "AC1", "std": "SD", "skew": "Skew.", "kurt": "Kurt.", "fd": "FD"}

# Value ranges to match your delta plotting
VMAP = {
    "sm": {
        "vmin": [-0.50, -0.04, -3,  -20, -0.2],
        "vmax": [ 0.50,  0.04,  3,   20,  0.2],
    },
    "Et": {
        "vmin": [-0.5, -0.3,  -3,  -15, -0.2],
        "vmax": [ 0.5,  0.3,   3,   15,  0.2],
    },
    "precip": {
        "vmin": [-0.5,  -15,   -3,  -20, -0.05],
        "vmax": [ 0.5,   15,    3,   20,  0.05],
    },
}


def _get_meanchange_field(ds: xr.Dataset, base: str, stat: str) -> xr.DataArray:
    """
    Try common mean-change naming conventions for a given base variable and stat.
    Returns the DataArray if found, else raises a clear KeyError.
    """
    candidates = [
        f"{base}_{stat}_mean_change",
        f"{base}_{stat}_meanchange",
        f"{base}_mean_change_{stat}",
        f"{base}_meanchange_{stat}",
    ]
    for name in candidates:
        if name in ds:
            return ds[name]
    # Helpful error: show nearby matches
    nearby = [v for v in ds.data_vars if base in v and stat in v]
    raise KeyError(
        f"Could not find mean-change field for base='{base}', stat='{stat}'. "
        f"Tried {candidates}. Nearby matches: {nearby[:10]}"
    )

def plot_meanchange(ds, var_name, out_path): 
    # Output dir
    outdir = out_path if out_path.endswith(os.sep) else out_path + os.sep
    os.makedirs(outdir, exist_ok=True)

    # Colormap
    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    # Choose vmin/vmax sets to mirror your delta plots
    if var_name not in VMAP:
        # Fallback: use precip ranges if unknown, else adjust as needed
        rng = VMAP["precip"]
    else:
        rng = VMAP[var_name]

    vmins = rng["vmin"]
    vmaxs = rng["vmax"]

    for i, stat in enumerate(INDICATOR_STATS):
        label = LABELS[stat]
        var = _get_meanchange_field(ds, var_name, stat)  # e.g., sm_ac1_mean_change

        fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, color='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        sc = var.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vmins[i],
            vmax=vmaxs[i],
            cmap=cmap,
            add_colorbar=False,
            rasterized=True
        )

        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # Save map
        outfile = f"{outdir}{var_name}_meanchange_{label.lower()}.png"
        plt.savefig(outfile, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        cbar.set_label(label)

        # Save standalone colorbar (SVG)
        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))
        norm = plt.Normalize(vmin=vmins[i], vmax=vmaxs[i])
        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb, orientation='horizontal')
        cb1.set_label(label)
        cb1.ax.tick_params(labelsize=8)
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)

        colorbar_outfile = f"{outdir}colorbar_{var_name}_meanchange_{label.lower()}.svg"
        plt.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f"Saved to: {outfile}")

def main(): 
    warnings.filterwarnings('ignore')

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'

    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--variable',     required=True)
    p.add_argument('--output_path',  required=True)
    args = p.parse_args()

    ds_path = args.dataset_path
    var_name = args.variable
    out_path = args.output_path

    # Open Zarr correctly
    ds = xr.open_zarr(ds_path)

    # Optional masking for precip (move AFTER opening ds so coords are known)
    if var_name == "precip":
        mask_ds = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")

        mask_ds = mask_ds.rio.write_crs("EPSG:4326")
        mask_ds = mask_ds.assign_coords(longitude=(((mask_ds.longitude + 180) % 360) - 180)).sortby("longitude")
        mask_ds.rio.set_spatial_dims("longitude", "latitude", inplace=True)

        #mask_on_ds = mask_ds.interp(
        ds_mask = mask_ds.interp(
        longitude=ds["lon"],  
        latitude=ds["lat"],
        method="nearest",
        )
 
        # Apply mask: keep only where lsm > 0.7
        mask = ds_mask["lsm"] > 0.7
        ds = ds.where(mask)

    plot_meanchange(ds, var_name, out_path)
    print('Done!')

if __name__=='__main__':
    main()