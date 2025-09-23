import re
import os
import glob
import scipy
import pickle
import warnings 
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

import argparse

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




""" 
Script to plot and save figures of the year of detected abrupt shift, 
and a combined, bivarate map of the absolute and total difference in the variable 
before and after the detected shift. 

Arguments are: 
    dataset_path: Path to the changepoint output file 
    var: Name of the variable in the changepoint output file
    out_dir: Path to the directory to save the file to 

E.g. 
    python 03a-plot_changepoints.py --dataset_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr' --var 'sm' --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_3'
    python 03a-plot_changepoints.py --dataset_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr' --var 'Et' --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_3'
    python 03a-plot_changepoints.py --dataset_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr' --var 'precip' --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_3'
 
"""



def adjust_colormap(cmap, minval=0.0, maxval=1.0, saturation_scale=0.5, n=256):
    """
    Truncate and desaturate a colormap by adjusting saturation.
    """
    # Sample original cmap
    sampled = cmap(np.linspace(minval, maxval, n))
    
    # Desaturate each color
    desat_colors = []
    for r, g, b, a in sampled:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s *= saturation_scale  # Reduce sat
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        desat_colors.append((r_new, g_new, b_new, a))

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'desaturated({cmap.name})',
        desat_colors
    )
    return new_cmap

def truncate_colormap(cmap, minval=0.0, maxval=0.85, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name})",
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def create_bivariate_color_log(prop_array, abs_array,
                           vmin_prop=0, vmax_prop=2, 
                           vmin_abs=0, vmax_abs=2,
                           cmap_prop='RdBu', cmap_abs=sn.light_palette('#AA180E', as_cmap = True)):
    """
    Take 2D numpy arrays (one for prop and one for abs difference),
    normalize them to [0,1] based on provided limits, extract colors using two 
    separate colormaps, and blend the colors-
    """
    # Normalize 
    norm_prop = np.clip((prop_array - vmin_prop) / (vmax_prop - vmin_prop), 0, 1)
    norm_abs = np.log1p(np.clip(abs_array, 0, vmax_abs)) / np.log1p(vmax_abs)
    
    # Get the cmap
    cmap1 = plt.get_cmap(cmap_prop)
    cmap2 = plt.get_cmap(cmap_abs)
    
    # Map normalized values to RGBA colors.
    colors_prop = cmap1(norm_prop)
    colors_abs  = cmap2(norm_abs)

    weight_abs = 0.6
    weight_prop = 1.0 - weight_abs

    blended_rgb = (weight_prop * colors_prop[..., :3] + weight_abs * colors_abs[..., :3])
    blended_rgb = np.clip(blended_rgb, 0, 1)
    gamma = 1.2  # lower = more contrast
    blended_rgb = blended_rgb ** gamma

    # Add alpha (set to 1 everywhere)
    blended_rgba = np.concatenate([blended_rgb, np.ones(blended_rgb.shape[:-1] + (1,))], axis=-1)
    
    return blended_rgba


def plot_year_chp(ds, var_name, outdir, test_name):

    """ 
    Plot and save maps and colourbar of the year of changepoint detected. 
    Choose from test: 'pettitt', 'stc', and 'var'
    
    """

    if test_name == 'pettitt': 
        cp_type = 'pettitt_cp' 
        pval_type = 'pettitt_pval'

    elif test_name == 'stc':
        cp_type = 'strucchange_bp'
        pval_type = 'Fstat_pval'

    elif test_name == 'var':
        cp_type = 'bp_var'
        pval_type = 'pval_var'

    start_year = 2000
    weeks_per_year = 52.1775
    tick_indices = np.arange(0, 1200, 200)

    ## Set up plot
    fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})

    ax.add_feature(cfeature.LAND, facecolor='white', linewidth = 0.5, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth = 0.25)
    ax.set_global()
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')


    tick_labels = [f"{int(start_year + idx / weeks_per_year)}" for idx in tick_indices]
    orig_cmap = plt.get_cmap("cubehelix")
    custom_cmap = adjust_colormap(orig_cmap, minval=0.0, maxval=0.9, saturation_scale=0.7)


    ## Plot and save map 
    ds_cp_valid = ds[f'{cp_type}'].where((ds[f'{pval_type}'] < 0.05) & (ds[f'{cp_type}'] > 0))
    sc = ds_cp_valid.plot(ax = ax, 
                        cmap = custom_cmap,
                        vmin = 0, vmax = 1200,
                        transform = ccrs.PlateCarree(),
                        add_colorbar = False)
    

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.7, label='Year of abrupt shift', pad = 0.05)
    cbar.set_ticks(tick_indices)
    cbar.set_ticklabels(tick_labels)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    """ ax.gridlines(color='black', alpha=0.5, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5) """


    fig.savefig(f"{outdir}/{var_name}_{test_name}_cpyear_map.png", dpi=300, bbox_inches='tight', facecolor='white')

    ## Plot and save colourbar 

    cbar_ax = cbar.ax  
    cbar_fig = plt.figure(figsize=(3.5, 0.3))  
    new_cbar_ax = cbar_fig.add_axes([0.05, 0.5, 0.9, 0.4])  

    plt.colorbar(sc, cax=new_cbar_ax, orientation="horizontal", label='Year of abrupt shift',
                ticks=tick_indices)
    new_cbar_ax.set_xticklabels(tick_labels)

    cbar_fig.savefig(f"{outdir}/{var_name}_{test_name}_cpyear_cbar.svg", format='svg', bbox_inches='tight')
    plt.close(cbar_fig)



def plot_prop_abs_change(ds, var_name, outdir, test_name):

    if test_name == 'pettitt': 
        cp_type = 'pettitt_cp'
        prop_type = 'prop_pettitt' 
        diff_type = 'diff_pettitt'
        pval_type = 'pettitt_pval'

    elif test_name == 'stc':
        cp_type = 'strucchange_bp'
        prop_type = 'prop_stc' 
        diff_type = 'diff_stc'
        pval_type = 'Fstat_pval'

    elif test_name == 'var':
        cp_type = 'bp_var'
        prop_type = 'prop_var' 
        diff_type = 'diff_var'
        pval_type = 'pval_var'


    # masks: only significant breakpoints
    prop = ds[f'{prop_type}'].where((ds[f'{pval_type}'] < 0.05) & (ds[f'{cp_type}'] > 0)).values
    abs_diff = ds[f'{diff_type}'].where((ds[f'{pval_type}'] < 0.05) & (ds[f'{cp_type}'] > 0)).values

    # coordinates for imshow 
    lon = ds['lon'].values
    lat = ds['lat'].values
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # combined bivariate colors
    combined_colors = create_bivariate_color_log(
        prop, abs_diff,
        vmin_prop=0.75, vmax_prop=1.25,
        vmin_abs=0,   vmax_abs=0.01,
        cmap_prop='RdBu',
        cmap_abs=truncate_colormap(sn.light_palette("#ab7171ff", as_cmap=True), 0.0, 0.8)
    )

    # MAP FIGURE 
    fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})
    ax.add_feature(cfeature.LAND, facecolor='white', linewidth=0.5, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.set_global()
    # Clip out antarctica here if you want
    # ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # remove border
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')

    # alpha where both layers are valid
    alpha_mask = np.isfinite(prop) & np.isfinite(abs_diff)
    combined_colors[..., 3] = alpha_mask.astype(float)

    # draw once (background) then final image on top
    ax.imshow(np.ones_like(combined_colors),
              extent=[lon_min, lon_max, lat_min, lat_max],
              transform=ccrs.PlateCarree(), origin='upper', zorder=0)

    ax.imshow(combined_colors, origin='upper',
              extent=[lon_min, lon_max, lat_min, lat_max],
              interpolation='none', transform=ccrs.PlateCarree(), zorder=1)

    # no ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    #  SAVE MAP PNG 
    map_path = f"{outdir}/{var_name}_{test_name}_prop_abs_map.png"
    fig.savefig(map_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)

    # Legend
    grid_size = 100
    prop_vals = np.linspace(0.75, 1.25, grid_size)  # y-axis
    abs_vals  = np.linspace(0.0, 0.01, grid_size)   # x-axis
    prop_grid, abs_grid = np.meshgrid(prop_vals, abs_vals, indexing='ij')

    legend_colors = create_bivariate_color_log(
        prop_grid, abs_grid,
        vmin_prop=0.75, vmax_prop=1.25,
        vmin_abs=0.0,   vmax_abs=0.01,
        cmap_prop='RdBu',
        cmap_abs=truncate_colormap(sn.light_palette("#ab7171ff", as_cmap=True), 0.0, 0.8)
    )

    fig_leg, ax_leg = plt.subplots(figsize=(3, 3))
    ax_leg.imshow(legend_colors, origin='lower',
                  extent=[0.0, 0.01, 0.75, 1.25],
                  aspect='auto')

    ax_leg.set_xlabel('Absolute difference')
    ax_leg.set_ylabel('Proportional difference')
    ax_leg.set_xticks([0.0, 0.005, 0.01])
    ax_leg.set_yticks([0.75, 1.0, 1.25])

    # clean frame
    for spine in ax_leg.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # SAVE LEGEND SVG 
    legend_path = f"{outdir}/{var_name}_{test_name}_prop_abs_legend.svg"
    fig_leg.savefig(legend_path, dpi = 300, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig_leg)

    return map_path, legend_path




def main(): 

    warnings.filterwarnings('ignore')

    sn.set_style("white")

    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'

    parser = argparse.ArgumentParser(description='Plot changepoint detection results of a dataset.') 
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the changepoint output.') 
    parser.add_argument('--var', type=str, required=True, help='Variable name to process (e.g., Et, precip, sm).')
    parser.add_argument('--out_dir', type = str, required=True, help='Path save output.')
    
    args = parser.parse_args()
    ds_path = args.dataset_path
    var_name = args.var 
    outdir = args.out_dir

    ds = xr.open_dataset(ds_path)

    # Land mask for precipitation 
    if var_name == "precip":
        ds_mask = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")

        ds_mask = ds_mask.rio.write_crs('EPSG:4326')
        ds_mask = ds_mask.assign_coords(longitude=(((ds_mask.longitude + 180) % 360) - 180)).sortby('longitude')
        ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)
        
        ds_mask = ds_mask.interp(
            longitude=ds.lon,   
            latitude=ds.lat,
            method="nearest"
        )
        
        mask = ds_mask["lsm"] > 0.7
        ds = ds.where(mask)

    print('--- Saving maps of changepoint year ---')
    plot_year_chp(ds, var_name, outdir, test_name = 'pettitt')
    plot_year_chp(ds, var_name, outdir, test_name = 'stc')
    plot_year_chp(ds, var_name, outdir, test_name = 'var')

    print('--- Saving maps of changepoint ---')
    plot_prop_abs_change(ds, var_name, outdir, test_name = 'pettitt')
    plot_prop_abs_change(ds, var_name, outdir, test_name = 'stc')
    plot_prop_abs_change(ds, var_name, outdir, test_name = 'var')


if __name__ == '__main__':

    main()



