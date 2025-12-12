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

### Formatting helpers 

MM_TO_IN = 1.0 / 25.4

TARGET_W_MM = 80.44     # width in mm
TARGET_H_MM = 39.760    # height in mm

FIGSIZE_SMALL = (
    TARGET_W_MM * MM_TO_IN,
    TARGET_H_MM * MM_TO_IN
)

REF_FIG_WIDTH_IN = 20.0

# Bivariate legend size (for prop/abs legend)
BIV_LEG_W_MM = 14.8
BIV_LEG_H_MM = 14.6

# 1D legend size (for year-of-breakpoint colorbar, similar to FD legend)
YEAR_LEG_W_MM = 17.5
YEAR_LEG_H_MM = 5.0


def scaled_lw(base_lw: float, ax) -> float:
    """Scale a base linewidth by figure width so small figures get thinner lines."""
    fig_w = ax.figure.get_size_inches()[0]
    scale = fig_w / REF_FIG_WIDTH_IN
    return base_lw * scale


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def land_mask_bool_like(ds: xr.Dataset) -> xr.DataArray:
    """Boolean land mask on ds grid using Natural Earth regions."""
    land_regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    lm = land_regions.mask(ds)
    return (~lm.isnull())


def add_base_map(ax):
    """Base land + coastlines styled like the non-five-panel plots."""
    lw_coast = scaled_lw(2, ax)
    ax.add_feature(
        cfeature.LAND,
        facecolor='white',
        linewidth=0.0,
        edgecolor='none',
        zorder=0
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=lw_coast,
        zorder=10
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def add_nodata_mask(
    ax, mask_da, lon, lat, extent,
    min_lat=-60.0,
    grey_rgba=(0.94, 0.94, 0.94, 1.0),
    outline_alpha=1,
    z_fill=2, z_tex=2.6, z_outline=3,
    fill_beyond_coverage=True
):
    """
    Light-grey fill where mask==True, then overlay one-way black hatch
    using contourf. Thin semi-transparent outline on top.

    This is copied from the 02a script to keep style identical.
    """
    grey_da = (mask_da & (mask_da["lat"] > min_lat))
    grey_vals = grey_da.transpose('lat', 'lon').fillna(False).astype(int).values

    cmap_mask = ListedColormap([(0, 0, 0, 0), grey_rgba])
    ax.imshow(
        grey_vals, origin='upper', extent=extent,
        transform=ccrs.PlateCarree(), interpolation='nearest',
        cmap=cmap_mask, zorder=z_fill
    )


    hatch_coll = ax.contourf(
        lon, lat, grey_vals,
        levels=[0.5, 1.5],
        colors='none',
        hatches=['////'],
        transform=ccrs.PlateCarree(),
        zorder=z_tex
    )
    for coll in hatch_coll.collections:
        coll.set_facecolor('none')
        coll.set_edgecolor('black')
        coll.set_linewidth(0.0)

    outline = ax.contour(
        lon, lat, grey_vals,
        levels=[0.5],
        colors='black',
        transform=ccrs.PlateCarree(),
        zorder=z_outline
    )
    for lc in outline.collections:
        lc.set_linewidth(scaled_lw(0.5, ax))
        lc.set_alpha(outline_alpha)
        lc.set_antialiased(True)

    if not fill_beyond_coverage:
        return

    dlat = float(np.median(np.abs(np.diff(lat))))
    lat_max_cov = float(lat.max())

    cap_overlap = 0.01
    cap_min_lat = min(90.0, lat_max_cov - cap_overlap)
    if cap_min_lat >= 90.0:
        return

    land_path = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_reader = shpreader.Reader(land_path)
    north_cap = box(-180.0, cap_min_lat, 180.0, 90.0)

    for geom in land_reader.geometries():
        inter = geom.intersection(north_cap)
        if inter.is_empty:
            continue

        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor=grey_rgba, edgecolor='none', zorder=z_fill
        )
        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black', hatch='////',
            linewidth=scaled_lw(1, ax), zorder=z_tex
        )
        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black',
            linewidth=scaled_lw(0.5, ax), alpha=outline_alpha, zorder=z_outline
        )


### Colour helpers 



def adjust_colormap(cmap, minval=0.0, maxval=1.0, saturation_scale=0.5, n=256):
    """
    Truncate and desaturate a colormap by adjusting saturation in HLS color space.
    """
    sampled = cmap(np.linspace(minval, maxval, n))

    desat_colors = []
    for r, g, b, a in sampled:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s *= saturation_scale
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
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def create_bivariate_color_log(
    prop_array, abs_array,
    vmin_prop=0, vmax_prop=2,
    vmin_abs=0, vmax_abs=2,
    cmap_prop='RdBu',
    cmap_abs=sn.light_palette("#5E5D5D", as_cmap=True)
):
    """
    Given two 2D numpy arrays (for proportion and absolute difference),
    normalize them to [0,1] based on provided limits, map to colors using two
    colormaps, and blend them.
    """
    norm_prop = np.clip((prop_array - vmin_prop) / (vmax_prop - vmin_prop), 0, 1)

    # log-scaled abs differences in [0,1]
    norm_abs = np.log1p(np.clip(abs_array, 0, vmax_abs)) / np.log1p(vmax_abs)

    cmap1 = plt.get_cmap(cmap_prop)
    cmap2 = plt.get_cmap(cmap_abs)

    colors_prop = cmap1(norm_prop)
    colors_abs = cmap2(norm_abs)

    weight_abs = 0.6
    weight_prop = 1.0 - weight_abs

    blended_rgb = (weight_prop * colors_prop[..., :3] +
                   weight_abs * colors_abs[..., :3])
    blended_rgb = np.clip(blended_rgb, 0, 1)

    # mild gamma for a bit more contrast
    gamma = 1.0
    blended_rgb = blended_rgb ** gamma

    blended_rgba = np.concatenate(
        [blended_rgb, np.ones(blended_rgb.shape[:-1] + (1,))],
        axis=-1
    )
    return blended_rgba


# -------------------------------------------------------------------
# Plot year
# -------------------------------------------------------------------

def plot_year_chp(ds, var_name, outdir, test_name, land_mask_bool, mask_da):
    """
    Plot and save maps and discrete colourbar of the year of changepoint detected.
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
    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    ensure_dir(outdir)

    lon = ds['lon'].values
    lat = ds['lat'].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    start_year = 2000
    weeks_per_year = 52.1775
    tick_indices = np.arange(0, 1200, 200)
    tick_labels = [f"{int(start_year + idx / weeks_per_year)}" for idx in tick_indices]

    orig_cmap = plt.get_cmap("cubehelix")
  
    custom_cmap = adjust_colormap(orig_cmap, minval=0.1, maxval=0.9, saturation_scale=0.7)


    ds_cp_valid = ds[cp_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0))

    # --- MAP  ---
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())
    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')

    sc = ds_cp_valid.plot(
        ax=ax,
        cmap=custom_cmap,
        vmin=0,
        vmax=1200,
        transform=ccrs.PlateCarree(),
        add_colorbar=False
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

  
    add_nodata_mask(
        ax, mask_da, lon, lat, extent,
        min_lat=-60.0,
        outline_alpha=1,
        fill_beyond_coverage=True
    )

    map_path = os.path.join(outdir, f"{var_name}_{test_name}_cpyear_map.png")
        
    fig.savefig(
        map_path,
        dpi=450,
        bbox_inches='tight',
        facecolor='white'
    )
    
    plt.close(fig)


    norm = mpl_colors.Normalize(vmin=0, vmax=1200)

    fig_cb = plt.figure(
        figsize=(YEAR_LEG_W_MM * MM_TO_IN, YEAR_LEG_H_MM * MM_TO_IN)
    )
    ax_cb = fig_cb.add_axes([0.15, 0.4, 0.7, 0.4])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    cb = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label('Year of abrupt shift', fontsize=5)
    cb.ax.set_xticks(tick_indices)
    cb.ax.set_xticklabels(tick_labels, fontsize=4)
    cb.outline.set_linewidth(0.3333)

    cbar_path = os.path.join(outdir, f"{var_name}_{test_name}_cpyear_cbar.svg")
    fig_cb.savefig(
        cbar_path,
        format='svg',
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        transparent=True
    )
    plt.close(fig_cb)


# -------------------------------------------------------------------
# Plot proportional & absolute change (bivariate)
# -------------------------------------------------------------------


def plot_prop_abs_change(ds, var_name, outdir, test_name, land_mask_bool, mask_da):

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

    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    ensure_dir(outdir)

    # --- masks: only significant breakpoints ---

    prop = ds[prop_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0)).values
    abs_diff = ds[diff_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0)).values

    lon = ds['lon'].values
    lat = ds['lat'].values
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    extent = [lon_min, lon_max, lat_min, lat_max]

    # --- combined bivariate colors ---

    abs_cmap = truncate_colormap(
        sn.light_palette("#666666ff", as_cmap=True),
        minval=0.2,
        maxval=1.0
    )

    combined_colors = create_bivariate_color_log(
        prop, abs_diff,
        vmin_prop=0.75, vmax_prop=1.25,                         ##### Edit this to match vmin and vmax prop
        vmin_abs=0.0, vmax_abs=0.01,                            ##### Edit this to match vmin and vmax abs
        cmap_prop='RdBu',
        cmap_abs=abs_cmap
    )

  
    alpha_mask = np.isfinite(prop) & np.isfinite(abs_diff) & land_mask_bool.values
    combined_colors[..., 3] = alpha_mask.astype(float)

  
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())
    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')

    ax.imshow(
        np.ones_like(combined_colors),
        extent=extent,
        transform=ccrs.PlateCarree(),
        origin='upper',
        zorder=0
    )

    ax.imshow(
        combined_colors,
        origin='upper',
        extent=extent,
        interpolation='none',
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

  
    add_nodata_mask(
        ax, mask_da, lon, lat, extent,
        min_lat=-60.0,
        outline_alpha=1,
        fill_beyond_coverage=True
    )

    map_path = os.path.join(outdir, f"{var_name}_{test_name}_prop_abs_map.png")
    fig.savefig(
        map_path,
        dpi=450,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close(fig)

    # --- DISCRETE LEGEND  ---
    n_bins_prop = 5
    n_bins_abs = 5

    prop_edges = np.linspace(0.75, 1.25, n_bins_prop + 1)
    abs_edges = np.linspace(0.0, 0.01, n_bins_abs + 1)

    prop_centers = 0.5 * (prop_edges[:-1] + prop_edges[1:])
    abs_centers = 0.5 * (abs_edges[:-1] + abs_edges[1:])

    prop_grid, abs_grid = np.meshgrid(prop_centers, abs_centers, indexing='ij')

    legend_colors = create_bivariate_color_log(
        prop_grid, abs_grid,
        vmin_prop=0.75, vmax_prop=1.25,
        vmin_abs=0.0, vmax_abs=0.01,
        cmap_prop='RdBu',
        cmap_abs=abs_cmap
    )

    fig_leg = plt.figure(
        figsize=(BIV_LEG_W_MM * MM_TO_IN, BIV_LEG_H_MM * MM_TO_IN)
    )
    ax_leg = fig_leg.add_axes([0.35, 0.30, 0.6, 0.6])

    ax_leg.imshow(
        legend_colors,
        origin='lower',
        extent=[0.0, 0.01, 0.75, 1.25],
        interpolation='nearest'
    )
    ax_leg.set_aspect('auto') 
    
    ax_leg.set_xlabel('Absolute difference', fontsize=6)
    ax_leg.set_ylabel('Proportional difference', fontsize=6)
    ax_leg.set_xticks([0.0, 0.005, 0.01])
    ax_leg.set_yticks([0.75, 1.0, 1.25])
    ax_leg.tick_params(axis="both", labelsize=5)

    for spine in ax_leg.spines.values():
        spine.set_visible(True)

    legend_path = os.path.join(outdir, f"{var_name}_{test_name}_prop_abs_legend.svg")
    fig_leg.savefig(
        legend_path,
        dpi=450,
        format='svg',
        bbox_inches=None,
        pad_inches=0.0,
        transparent=True
    )
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

    ds_for_mask = xr.open_dataset(f'/mnt/data/romi/output/paper_1/output_{var_name}_final/out_{var_name}_kt.zarr')
    land_mask_bool = land_mask_bool_like(ds_for_mask)

    ac1_name = f"{var_name}_ac1_kt"
    if ac1_name in ds_for_mask:
        mask_da = ds_for_mask[ac1_name].isnull() & land_mask_bool
    else:
        # fallback: no no-data masking beyond land mask
        mask_da = land_mask_bool.copy(deep=False)
        mask_da[:] = False

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



