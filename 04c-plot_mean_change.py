
import os
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


from pathlib import Path
from utils.config import load_config, cfg_path, cfg_get

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

def _get_meanchange_field(ds, base, stat):
    candidates = [
        f"{base}_{stat}_mean_change",
        f"{base}_{stat}_meanchange",
        f"{base}_mean_change_{stat}",
        f"{base}_meanchange_{stat}",
    ]
    for name in candidates:
        if name in ds:
            return ds[name]
    return None


def _mask_precip_to_land(ds: xr.Dataset, cfg) -> xr.Dataset:
    """Mask precip dataset to land (lsm > 0.7) on ds lon/lat grid."""
    mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
    mask_ds = xr.open_dataset(mask_path, engine="cfgrib")

    mask_ds = mask_ds.rio.write_crs("EPSG:4326")
    mask_ds = mask_ds.assign_coords(
        longitude=(((mask_ds.longitude + 180) % 360) - 180)
    ).sortby("longitude")
    mask_ds.rio.set_spatial_dims("longitude", "latitude", inplace=True)

    ds_mask = mask_ds.interp(
        longitude=ds["lon"],
        latitude=ds["lat"],
        method="nearest",
    )

    land = (ds_mask["lsm"] > 0.7)

    # Rename dims/coords to match ds
    if "longitude" in land.dims:
        if "lon" in land.coords:
            land = land.swap_dims({"longitude": "lon"}).drop_vars("longitude")
        else:
            land = land.rename({"longitude": "lon"})
    if "latitude" in land.dims:
        if "lat" in land.coords:
            land = land.swap_dims({"latitude": "lat"}).drop_vars("latitude")
        else:
            land = land.rename({"latitude": "lat"})

    land = land.reindex_like(ds, method=None)
    return ds.where(land)


def plot_meanchange(ds, var_name, out_path):
    # Output dir
    outdir = out_path if out_path.endswith(os.sep) else out_path + os.sep
    os.makedirs(outdir, exist_ok=True)

    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    rng = VMAP[var_name] if var_name in VMAP else VMAP["precip"]
    vmins = rng["vmin"]
    vmaxs = rng["vmax"]

    for i, stat in enumerate(INDICATOR_STATS):
        label = LABELS[stat]
        var = _get_meanchange_field(ds, var_name, stat)

        if var is None:
            print(f"[warn] Missing mean-change field for {var_name} {stat}; skipping.")
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={"projection": ccrs.Robinson()})
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, color="black")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        sc = var.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vmins[i],
            vmax=vmaxs[i],
            cmap=cmap,
            add_colorbar=False,
            rasterized=True,
        )

        ax.gridlines(color="black", alpha=0.5, linestyle="--", linewidth=0.5)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # Save map
        outfile = f"{outdir}{var_name}_meanchange_{label.lower()}.png"
        plt.savefig(outfile, format="png", dpi=300, bbox_inches="tight", facecolor="white")

        # (Optional) also add a map colorbar if you ever want it later:
        # cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        # cbar.set_label(label)

        plt.close(fig)

        # Save separate colorbar SVG
        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))
        norm = plt.Normalize(vmin=vmins[i], vmax=vmaxs[i])
        cb1 = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            orientation="horizontal",
        )
        cb1.set_label(label)
        cb1.ax.tick_params(labelsize=8)
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)

        colorbar_outfile = f"{outdir}colorbar_{var_name}_meanchange_{label.lower()}.svg"
        plt.savefig(colorbar_outfile, format="svg", dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig_cb)

        print(f"Saved to: {outfile}")


def main():
    warnings.filterwarnings("ignore")

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    p = argparse.ArgumentParser()
    p.add_argument("--variable", required=True, help="sm, Et, precip")
    p.add_argument("--suffix", default=None,
                   help="Optional suffix for inferred dataset (e.g. breakpoint_stc).")
    p.add_argument("--dataset_path", default=None,
                   help="Optional override path to *_meanchange.zarr. If omitted, inferred from config + --variable + --suffix.")
    p.add_argument("--output_path", default=None,
                   help="Optional override output directory. If omitted, uses outputs_root/figures/meanchange/<variable>/")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))

    def _sfx(s):
        if not s:
            return ""
        s = str(s).strip()
        return s if s.startswith("_") else f"_{s}"

    default_ds = outputs_root / "zarr" / f"out_{args.variable}{_sfx(args.suffix)}_meanchange.zarr"
    ds_path = Path(args.dataset_path) if args.dataset_path else default_ds

    default_out = outputs_root / "figures" / "meanchange" / args.variable
    out_path = str(Path(args.output_path)) if args.output_path else str(default_out)

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    ds = xr.open_dataset(str(ds_path))

    if args.variable == "precip":
        ds = _mask_precip_to_land(ds, cfg)

    plot_meanchange(ds, args.variable, out_path)
    print("Done!")


if __name__ == "__main__":
    main()