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
Plots and saves deltas of the EWS. 

Inputs: 
    dataset_path: EWS file path to plot
    variable: Variable (e.g. sm, Et, precip)
    output_path: Directory to save figures to. Needs a / at the end

E.g.
    python 01b-plot_deltas.py --dataset_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr' --variable 'sm' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2'
    python 01b-plot_deltas.py --dataset_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr' --variable 'Et' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2'
    python 01b-plot_deltas.py --dataset_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr' --variable 'precip' --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_2'
"""

def plot_deltas(ds, var_name, out_path): 

    # List of variables and settings
    variables = [
            ds[f'{var_name}_delta_ac1'],
            ds[f'{var_name}_delta_std'],
            ds[f'{var_name}_delta_skew'],
            ds[f'{var_name}_delta_kurt'],
            ds[f'{var_name}_delta_fd']
        ]
    labels = ["AC1", "SD", "Skew.", "Kurt.", "FD"]

    # adjust as needed for plotting

    """ #soil moisture
    vmins = [-1.5, -0.075, -10, -35, -0.5]
    vmaxs = [1.5, 0.075, 10, 35, 0.5] """

    """ #transpration
    vmins = [-1.5, -1, -10, -35, -1]
    vmaxs = [1.5, 1, 10, 35, 1] """
    
    #precipitation 
    vmins = [-1, -30, -10, -35, -0.5]
    vmaxs = [1, 30, 10, 35, 0.5]

    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    # Output directory
    outdir = out_path

    for var, label, vmin, vmax in zip(variables, labels, vmins, vmaxs):
        fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})
        
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none',linewidth=0.5,zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, color = 'black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        sc = var.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            add_colorbar=False,
            rasterized=True  
        )

        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth = 0.5)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        cbar.set_label(label)

        # Save individually
        outfile = f"{outdir}{var_name}_deltas_{label.lower()}.png"
        plt.savefig(outfile, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)


        # Save the colorbar 

        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))  # Adjust size as needed

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            orientation='horizontal'
        )
        cb1.set_label(label)
        cb1.ax.tick_params(labelsize=8)

        # Remove surrounding whitespace
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)

        # Save as SVG (preferred for Inkscape) or PNG
        colorbar_outfile = f"{outdir}colorbar_{var_name}_deltas_{label.lower()}.svg"
        plt.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f'Saved to: {outfile}')

def main(): 

    warnings.filterwarnings('ignore')

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'



    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path',   required=True)
    p.add_argument('--variable',  required=True)
    p.add_argument('--output_path',  required=True)
    args = p.parse_args()

    ds_path = args.dataset_path
    var_name = args.variable
    out_path = args.output_path

    if var_name == "precip":
        ds_mask = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")

        # Detect coordinate names
        lon_name = None
        lat_name = None
        for cand in ["lon", "longitude", "x"]:
            if cand in ds.coords:
                lon_name = cand
                break
        for cand in ["lat", "latitude", "y"]:
            if cand in ds.coords:
                lat_name = cand
                break
        if lon_name is None or lat_name is None:
            raise ValueError(f"Could not detect lat/lon coordinate names in dataset {list(ds.coords)}")

        # Align mask to dataset grid
        ds_mask = ds_mask.interp(
            {lon_name: ds[lon_name], lat_name: ds[lat_name]},
            method="nearest"
        )

        # Apply mask: keep only where lsm > 0.7
        mask = ds_mask["lsm"] > 0.7
        ds = ds.where(mask)

    ds = xr.open_dataset(ds_path)
    plot_deltas(ds, var_name, out_path)

    print('Done!')


if __name__=='__main__':

    main()
