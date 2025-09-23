import os
import argparse
import warnings

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sn

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
Plots and saves Theil–Sen slope maps (per year) for AC1, SD, Skew, Kurt, FD.

Inputs:
  --dataset_path : path to merged *_ts.zarr produced by the Theil–Sen script
  --variable     : variable prefix used in that dataset (e.g. sm, Et, precip)
  --output_path  : directory to save images to (can be without trailing '/')
  --sig_only     : if set, mask non-significant pixels (ts_sig == 0)
  --auto_range   : if set (default), pick symmetric vmin/vmax from robust quantiles
  --q            : quantile for auto_range (default 0.995)

E.g.:
  python 04a-plot_theil_sen.py \
    --dataset_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_ts.zarr' \
    --variable 'sm' \
    --output_path '/mnt/data/romi/figures/paper_1/supplementary_final/supp_3' \
    --sig_only

"""

SUFFIXES = ["ac1", "std", "skew", "kurt", "fd"]
LABELS   = ["AC1", "SD", "Skew.", "Kurt.", "FD"]

def open_source_ds(path):
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup")) or
        os.path.exists(os.path.join(path, ".zmetadata"))
    ):
        return xr.open_zarr(path)
    return xr.open_dataset(path)

def robust_sym_limits(da, q = 0.995):
    """Symmetric limits around 0 for plotting."""
    try:
        a = float(np.abs(da).quantile(q, skipna=True))
    except Exception:
        # Fallback if quantile not available
        a = float(np.nanquantile(np.abs(da.values), q))
    if not np.isfinite(a) or a == 0.0:
        a = float(np.nanmax(np.abs(da.values)))
    if not np.isfinite(a) or a == 0.0:
        a = 1.0
    return -a, a

def plot_theilsen(ds, var_name, out_dir,
                  sig_only = False, auto_range = True, q= 0.995):
    
    os.makedirs(out_dir, exist_ok=True)

    # Build list of slope variables to plot
    to_plot = []
    for suf, lab in zip(SUFFIXES, LABELS):
        vname = f"{var_name}_{suf}_ts"
        if vname in ds.variables:
            to_plot.append((ds[vname], lab, suf))
        else:
            print(f"[warn] Missing variable in dataset: {vname} (skipping).")

    if not to_plot:
        raise RuntimeError("No Theil–Sen slope variables found for given prefix.")

    # PLot only significant slopes (CI excludes 0) if true
    if sig_only:
        for i, (da, lab, suf) in enumerate(to_plot):
            sig_name = f"{var_name}_{suf}_ts_sig"
            if sig_name in ds:
                to_plot[i] = (da.where(ds[sig_name] > 0), lab, suf)
            else:
                print(f"[warn] Missing significance var {sig_name}; cannot mask {lab}.")

    # Cmap
    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    # Manual ranges
    # # soil moisture 
    # vmins = [-0.05, -0.01, -0.5, -2.0, -0.02]
    # vmaxs = [ 0.05,  0.01,  0.5,  2.0,  0.02]
    # # transpiration 
    # vmins = [-0.05, -0.5, -0.5, -2.0, -0.05]
    # vmaxs = [ 0.05,  0.5,  0.5,  2.0,  0.05]
    # # precipitation 
    # vmins = [-0.03, -1.0, -0.5, -2.0, -0.03]
    # vmaxs = [ 0.03,  1.0,  0.5,  2.0,  0.03]

    for idx, (da, label, suf) in enumerate(to_plot):
        # Decide color limits
        if auto_range:
            vmin, vmax = robust_sym_limits(da, q=q)
        else:
            vmin, vmax = robust_sym_limits(da, q=q)

        # Figure
        fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, color='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        sc = da.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            add_colorbar=False,
            rasterized=True
        )

        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_title("")  
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.xaxis.set_ticks([]); ax.yaxis.set_ticks([])

        # Cbar
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        cbar.set_label(f"{label} slope (per year)")

        # Save
        base = f"{var_name}_ts_{label.lower()}"
        if sig_only:
            base += "_sig"
        outfile = os.path.join(out_dir, f"{base}.png")
        plt.savefig(outfile, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Save a cbar
        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb,
            orientation='horizontal'
        )
        cb1.set_label(f"{label} slope (per year)")
        cb1.ax.tick_params(labelsize=8)
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)

        colorbar_outfile = os.path.join(out_dir, f"colorbar_{base}.svg")
        plt.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f"Saved: {outfile}")

def main():
    warnings.filterwarnings('ignore')

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', required=True, help="Path to *_ts.zarr or NetCDF")
    p.add_argument('--variable',     required=True, help="Variable prefix (e.g., sm, Et, precip)")
    p.add_argument('--output_path',  required=True, help="Directory to save figures")
    p.add_argument('--sig_only',     action='store_true', help="Mask non-significant slopes")
    p.add_argument('--auto_range',   action='store_true', help="Auto symmetric color limits (robust)")
    p.add_argument('--q',            type=float, default=0.995, help="Quantile for auto_range (default 0.995)")
    args = p.parse_args()

    ds = open_source_ds(args.dataset_path)
    out_dir = args.output_path.rstrip("/")

    plot_theilsen(
        ds, args.variable, out_dir,
        sig_only=args.sig_only,
        auto_range=args.auto_range or True,  # default on
        q=args.q
    )
    print("Done!")

if __name__ == '__main__':
    main()