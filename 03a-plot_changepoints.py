#!/usr/bin/env python3
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
from shapely.geometry import box

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

from pathlib import Path
from utils.config import load_config, cfg_path, cfg_get


"""
Script to plot and save figures of the year of detected abrupt shift,
and a combined, bivariate map of the absolute and total difference in the variable
before and after the detected shift.

Config-integrated:
- If --dataset/--dataset_path is omitted: infer from config paths.outputs_root:
    outputs_root/zarr/out_<var><_suffix>_chp.zarr
- If --outdir/--out_dir is omitted: default to:
    outputs_root/figures/changepoints/<var>/
- For "no-data" hatching mask, by default it tries:
    outputs_root/zarr/out_<var><_suffix>_kt.zarr
  You can override with --kt-dataset, or fall back to config datasets.<var>.kt_zarr if present.

Examples (inferred):
    python 03a-plot_changepoints.py --var sm --config config.yaml
    python 03a-plot_changepoints.py --var Et --suffix breakpoint_stc --config config.yaml

Examples (override):
    python 03a-plot_changepoints.py --var sm --dataset /path/out_sm_chp.zarr --outdir /path/figure_3 --config config.yaml
"""


# ---------------- Formatting helpers ----------------

MM_TO_IN = 1.0 / 25.4

TARGET_W_MM = 80.44
TARGET_H_MM = 39.760

FIGSIZE_SMALL = (TARGET_W_MM * MM_TO_IN, TARGET_H_MM * MM_TO_IN)

REF_FIG_WIDTH_IN = 20.0

BIV_LEG_W_MM = 14.8
BIV_LEG_H_MM = 14.6

YEAR_LEG_W_MM = 17.5
YEAR_LEG_H_MM = 5.0


def _sfx(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).strip()
    return s if s.startswith("_") else f"_{s}"


def open_source_ds(path: str) -> xr.Dataset:
    """Open NetCDF or Zarr."""
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zmetadata"))
    ):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def scaled_lw(base_lw: float, ax) -> float:
    fig_w = ax.figure.get_size_inches()[0]
    scale = fig_w / REF_FIG_WIDTH_IN
    return base_lw * scale


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def land_mask_bool_like(ds: xr.Dataset) -> xr.DataArray:
    land_regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    lm = land_regions.mask(ds)
    return (~lm.isnull())


def add_base_map(ax):
    lw_coast = scaled_lw(2, ax)
    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        linewidth=0.0,
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=lw_coast,
        zorder=10,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def add_nodata_mask(
    ax,
    mask_da,
    lon,
    lat,
    extent,
    min_lat=-60.0,
    grey_rgba=(0.94, 0.94, 0.94, 1.0),
    outline_alpha=1,
    z_fill=2,
    z_tex=2.6,
    z_outline=3,
    fill_beyond_coverage=True,
):
    """
    Light-grey fill where mask==True, then overlay one-way black hatch.
    """
    grey_da = (mask_da & (mask_da["lat"] > min_lat))
    grey_vals = grey_da.transpose("lat", "lon").fillna(False).astype(int).values

    cmap_mask = ListedColormap([(0, 0, 0, 0), grey_rgba])
    ax.imshow(
        grey_vals,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        cmap=cmap_mask,
        zorder=z_fill,
    )

    hatch_coll = ax.contourf(
        lon,
        lat,
        grey_vals,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        transform=ccrs.PlateCarree(),
        zorder=z_tex,
    )
    for coll in hatch_coll.collections:
        coll.set_facecolor("none")
        coll.set_edgecolor("black")
        coll.set_linewidth(0.0)

    outline = ax.contour(
        lon,
        lat,
        grey_vals,
        levels=[0.5],
        colors="black",
        transform=ccrs.PlateCarree(),
        zorder=z_outline,
    )
    for lc in outline.collections:
        lc.set_linewidth(scaled_lw(0.5, ax))
        lc.set_alpha(outline_alpha)
        lc.set_antialiased(True)

    if not fill_beyond_coverage:
        return

    lat_max_cov = float(lat.max())
    cap_overlap = 0.01
    cap_min_lat = min(90.0, lat_max_cov - cap_overlap)
    if cap_min_lat >= 90.0:
        return

    land_path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    land_reader = shpreader.Reader(land_path)
    north_cap = box(-180.0, cap_min_lat, 180.0, 90.0)

    for geom in land_reader.geometries():
        inter = geom.intersection(north_cap)
        if inter.is_empty:
            continue

        ax.add_geometries(
            [inter],
            crs=ccrs.PlateCarree(),
            facecolor=grey_rgba,
            edgecolor="none",
            zorder=z_fill,
        )
        ax.add_geometries(
            [inter],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            hatch="////",
            linewidth=scaled_lw(1, ax),
            zorder=z_tex,
        )
        ax.add_geometries(
            [inter],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=scaled_lw(0.5, ax),
            alpha=outline_alpha,
            zorder=z_outline,
        )


# ---------------- Colour helpers ----------------

def adjust_colormap(cmap, minval=0.0, maxval=1.0, saturation_scale=0.5, n=256):
    sampled = cmap(np.linspace(minval, maxval, n))
    desat_colors = []
    for r, g, b, a in sampled:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s *= saturation_scale
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        desat_colors.append((r_new, g_new, b_new, a))

    return mcolors.LinearSegmentedColormap.from_list(
        f"desaturated({cmap.name})",
        desat_colors,
    )


def truncate_colormap(cmap, minval=0.0, maxval=0.85, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name})",
        cmap(np.linspace(minval, maxval, n)),
    )


def create_bivariate_color_log(
    prop_array,
    abs_array,
    vmin_prop=0,
    vmax_prop=2,
    vmin_abs=0,
    vmax_abs=2,
    cmap_prop="RdBu",
    cmap_abs=sn.light_palette("#5E5D5D", as_cmap=True),
):
    norm_prop = np.clip((prop_array - vmin_prop) / (vmax_prop - vmin_prop), 0, 1)
    norm_abs = np.log1p(np.clip(abs_array, 0, vmax_abs)) / np.log1p(vmax_abs)

    cmap1 = plt.get_cmap(cmap_prop)
    cmap2 = plt.get_cmap(cmap_abs)

    colors_prop = cmap1(norm_prop)
    colors_abs = cmap2(norm_abs)

    weight_abs = 0.6
    weight_prop = 1.0 - weight_abs

    blended_rgb = (
        weight_prop * colors_prop[..., :3] +
        weight_abs * colors_abs[..., :3]
    )
    blended_rgb = np.clip(blended_rgb, 0, 1)

    gamma = 1.0
    blended_rgb = blended_rgb ** gamma

    blended_rgba = np.concatenate(
        [blended_rgb, np.ones(blended_rgb.shape[:-1] + (1,))],
        axis=-1,
    )
    return blended_rgba


# -------------------------------------------------------------------
# Plot year
# -------------------------------------------------------------------

def plot_year_chp(ds, var_name, outdir, test_name, land_mask_bool, mask_da):
    if test_name == "pettitt":
        cp_type = "pettitt_cp"
        pval_type = "pettitt_pval"
    elif test_name == "stc":
        cp_type = "strucchange_bp"
        pval_type = "Fstat_pval"
    elif test_name == "var":
        cp_type = "bp_var"
        pval_type = "pval_var"
    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    ensure_dir(outdir)

    lon = ds["lon"].values
    lat = ds["lat"].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    start_year = 2000
    weeks_per_year = 52.1775
    tick_indices = np.arange(0, 1200, 200)
    tick_labels = [f"{int(start_year + idx / weeks_per_year)}" for idx in tick_indices]

    orig_cmap = plt.get_cmap("cubehelix")
    custom_cmap = adjust_colormap(orig_cmap, minval=0.1, maxval=0.9, saturation_scale=0.7)

    ds_cp_valid = ds[cp_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0))

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={"projection": ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())
    fig.patch.set_facecolor("white")
    fig.set_facecolor("white")

    ds_cp_valid.plot(
        ax=ax,
        cmap=custom_cmap,
        vmin=0,
        vmax=1200,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
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
        fill_beyond_coverage=True,
    )

    map_path = os.path.join(outdir, f"{var_name}_{test_name}_cpyear_map.png")
    fig.savefig(map_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    norm = mpl_colors.Normalize(vmin=0, vmax=1200)

    fig_cb = plt.figure(figsize=(YEAR_LEG_W_MM * MM_TO_IN, YEAR_LEG_H_MM * MM_TO_IN))
    ax_cb = fig_cb.add_axes([0.15, 0.4, 0.7, 0.4])

    smap = mpl.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    smap.set_array([])

    cb = plt.colorbar(smap, cax=ax_cb, orientation="horizontal")
    cb.set_label("Year of abrupt shift", fontsize=5)
    cb.ax.set_xticks(tick_indices)
    cb.ax.set_xticklabels(tick_labels, fontsize=4)
    cb.outline.set_linewidth(0.3333)

    cbar_path = os.path.join(outdir, f"{var_name}_{test_name}_cpyear_cbar.svg")
    fig_cb.savefig(
        cbar_path,
        format="svg",
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig_cb)


# -------------------------------------------------------------------
# Plot proportional & absolute change (bivariate)
# -------------------------------------------------------------------

def plot_prop_abs_change(ds, var_name, outdir, test_name, land_mask_bool, mask_da):
    if test_name == "pettitt":
        cp_type = "pettitt_cp"
        prop_type = "prop_pettitt"
        diff_type = "diff_pettitt"
        pval_type = "pettitt_pval"
    elif test_name == "stc":
        cp_type = "strucchange_bp"
        prop_type = "prop_stc"
        diff_type = "diff_stc"
        pval_type = "Fstat_pval"
    elif test_name == "var":
        cp_type = "bp_var"
        prop_type = "prop_var"
        diff_type = "diff_var"
        pval_type = "pval_var"
    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    ensure_dir(outdir)

    prop = ds[prop_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0)).values
    abs_diff = ds[diff_type].where((ds[pval_type] < 0.05) & (ds[cp_type] > 0)).values

    lon = ds["lon"].values
    lat = ds["lat"].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    abs_cmap = truncate_colormap(
        sn.light_palette("#666666ff", as_cmap=True),
        minval=0.2,
        maxval=1.0,
    )

    combined_colors = create_bivariate_color_log(
        prop,
        abs_diff,
        vmin_prop=0.75,
        vmax_prop=1.25,
        vmin_abs=0.0,
        vmax_abs=0.01,
        cmap_prop="RdBu",
        cmap_abs=abs_cmap,
    )

    alpha_mask = np.isfinite(prop) & np.isfinite(abs_diff) & land_mask_bool.values
    combined_colors[..., 3] = alpha_mask.astype(float)

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={"projection": ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())
    fig.patch.set_facecolor("white")
    fig.set_facecolor("white")

    ax.imshow(
        np.ones_like(combined_colors),
        extent=extent,
        transform=ccrs.PlateCarree(),
        origin="upper",
        zorder=0,
    )

    ax.imshow(
        combined_colors,
        origin="upper",
        extent=extent,
        interpolation="none",
        transform=ccrs.PlateCarree(),
        zorder=1,
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
        fill_beyond_coverage=True,
    )

    map_path = os.path.join(outdir, f"{var_name}_{test_name}_prop_abs_map.png")
    fig.savefig(map_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Legend (discrete)
    n_bins_prop = 5
    n_bins_abs = 5

    prop_edges = np.linspace(0.75, 1.25, n_bins_prop + 1)
    abs_edges = np.linspace(0.0, 0.01, n_bins_abs + 1)

    prop_centers = 0.5 * (prop_edges[:-1] + prop_edges[1:])
    abs_centers = 0.5 * (abs_edges[:-1] + abs_edges[1:])

    prop_grid, abs_grid = np.meshgrid(prop_centers, abs_centers, indexing="ij")

    legend_colors = create_bivariate_color_log(
        prop_grid,
        abs_grid,
        vmin_prop=0.75,
        vmax_prop=1.25,
        vmin_abs=0.0,
        vmax_abs=0.01,
        cmap_prop="RdBu",
        cmap_abs=abs_cmap,
    )

    fig_leg = plt.figure(figsize=(BIV_LEG_W_MM * MM_TO_IN, BIV_LEG_H_MM * MM_TO_IN))
    ax_leg = fig_leg.add_axes([0.35, 0.30, 0.6, 0.6])

    ax_leg.imshow(
        legend_colors,
        origin="lower",
        extent=[0.0, 0.01, 0.75, 1.25],
        interpolation="nearest",
    )
    ax_leg.set_aspect("auto")

    ax_leg.set_xlabel("Absolute difference", fontsize=6)
    ax_leg.set_ylabel("Proportional difference", fontsize=6)
    ax_leg.set_xticks([0.0, 0.005, 0.01])
    ax_leg.set_yticks([0.75, 1.0, 1.25])
    ax_leg.tick_params(axis="both", labelsize=5)

    for spine in ax_leg.spines.values():
        spine.set_visible(True)

    legend_path = os.path.join(outdir, f"{var_name}_{test_name}_prop_abs_legend.svg")
    fig_leg.savefig(
        legend_path,
        dpi=450,
        format="svg",
        bbox_inches=None,
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig_leg)

    return map_path, legend_path


# ---------------- Main ----------------

def main():
    warnings.filterwarnings("ignore")

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    p = argparse.ArgumentParser(description="Plot changepoint detection results of a dataset.")
    p.add_argument("--var", type=str, required=True, help="Variable name (Et, precip, sm).")
    p.add_argument(
        "--suffix",
        default=None,
        help="Optional suffix used for inferred dataset paths (e.g. breakpoint_stc).",
    )
    # Kept for  backward compatibility
    p.add_argument(
        "--dataset",
        "--dataset_path",
        dest="dataset_path",
        default=None,
        help="Optional override path to the changepoint output (.zarr or netcdf). If omitted, inferred from config + --var + --suffix.",
    )
    p.add_argument(
        "--outdir",
        "--out_dir",
        dest="out_dir",
        default=None,
        help="Optional output directory. If omitted, uses outputs_root/figures/changepoints/<var>/",
    )
    p.add_argument(
        "--kt-dataset",
        dest="kt_dataset",
        default=None,
        help="Optional override path to *_kt.zarr for no-data masking. If omitted, inferred from outputs_root (or datasets.<var>.kt_zarr if present).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))
    zarr_dir = outputs_root / "zarr"

    # Infer changepoint dataset if not provided
    default_chp = zarr_dir / f"out_{args.var}{_sfx(args.suffix)}_chp.zarr"
    ds_path = Path(args.dataset_path) if args.dataset_path else default_chp

    # Infer outdir if not provided
    default_outdir = outputs_root / "figures" / "changepoints" / args.var
    outdir = Path(args.out_dir) if args.out_dir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Open changepoint dataset
    ds = open_source_ds(str(ds_path))
    var_name = args.var

    # Determine KT dataset path for mask
    kt_path = None
    if args.kt_dataset:
        kt_path = Path(args.kt_dataset)
    else:
        default_kt = zarr_dir / f"out_{args.var}{_sfx(args.suffix)}_kt.zarr"
        if default_kt.exists():
            kt_path = default_kt
        else:
            # Optional fallback to config 
            kt_cfg = None
            try:
                kt_cfg = cfg_get(cfg, f"datasets.{args.var}.kt_zarr", default=None)
            except Exception:
                kt_cfg = None
            if kt_cfg:
                kt_path = Path(kt_cfg)

    # Build land_mask_bool + mask_da (hatching)
    if kt_path is not None and kt_path.exists():
        ds_for_mask = open_source_ds(str(kt_path))
        land_mask_bool = land_mask_bool_like(ds_for_mask)

        ac1_name = f"{var_name}_ac1_kt"
        if ac1_name in ds_for_mask:
            mask_da = ds_for_mask[ac1_name].isnull() & land_mask_bool
        else:
            mask_da = land_mask_bool.copy(deep=False)
            mask_da[:] = False
        try:
            ds_for_mask.close()
        except Exception:
            pass
    else:
        # fallback: land mask only, no “no-data” hatch
        land_mask_bool = land_mask_bool_like(ds)
        mask_da = land_mask_bool.copy(deep=False)
        mask_da[:] = False

    # Land-sea mask for precipitation 
        mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
        ds_mask = xr.open_dataset(mask_path, engine="cfgrib")

        ds_mask = ds_mask.rio.write_crs("EPSG:4326")
        ds_mask = ds_mask.assign_coords(
            longitude=(((ds_mask.longitude + 180) % 360) - 180)
        ).sortby("longitude")
        ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)

        ds_mask = ds_mask.interp(
            longitude=ds.lon,
            latitude=ds.lat,
            method="nearest",
        )

        mask = ds_mask["lsm"] > 0.7
        ds = ds.where(mask)

    print("--- Saving maps of changepoint year ---")
    plot_year_chp(ds, var_name, str(outdir), test_name="pettitt", land_mask_bool=land_mask_bool, mask_da=mask_da)
    plot_year_chp(ds, var_name, str(outdir), test_name="stc",     land_mask_bool=land_mask_bool, mask_da=mask_da)
    plot_year_chp(ds, var_name, str(outdir), test_name="var",     land_mask_bool=land_mask_bool, mask_da=mask_da)

    print("--- Saving maps of changepoint (prop/abs bivariate) ---")
    plot_prop_abs_change(ds, var_name, str(outdir), test_name="pettitt", land_mask_bool=land_mask_bool, mask_da=mask_da)
    plot_prop_abs_change(ds, var_name, str(outdir), test_name="stc",     land_mask_bool=land_mask_bool, mask_da=mask_da)
    plot_prop_abs_change(ds, var_name, str(outdir), test_name="var",     land_mask_bool=land_mask_bool, mask_da=mask_da)


if __name__ == "__main__":
    main()