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
from utils.config import load_config, cfg_path, cfg_get
from pathlib import Path

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


def open_source_ds(path: str) -> xr.Dataset:
    # Per your preference: zarr opens fine with open_dataset
    return xr.open_dataset(path)


def robust_sym_limits(da, q: float = 0.995):
    """Symmetric limits around 0 for plotting."""
    try:
        a = float(np.abs(da).quantile(q, skipna=True))
    except Exception:
        a = float(np.nanquantile(np.abs(da.values), q))

    if not np.isfinite(a) or a == 0.0:
        a = float(np.nanmax(np.abs(da.values))) if np.isfinite(da.values).any() else 1.0
    if not np.isfinite(a) or a == 0.0:
        a = 1.0
    return -a, a


def plot_theilsen(ds, var_name: str, out_dir: str,
                  sig_only: bool = False, auto_range: bool = True, q: float = 0.995):

    os.makedirs(out_dir, exist_ok=True)

    to_plot = []
    for suf, lab in zip(SUFFIXES, LABELS):
        vname = f"{var_name}_{suf}_ts"
        if vname in ds.variables:
            to_plot.append((ds[vname], lab, suf))
        else:
            print(f"[warn] Missing variable in dataset: {vname} (skipping).")

    if not to_plot:
        raise RuntimeError("No Theil–Sen slope variables found for given prefix.")

    if sig_only:
        for i, (da, lab, suf) in enumerate(to_plot):
            sig_name = f"{var_name}_{suf}_ts_sig"
            if sig_name in ds:
                to_plot[i] = (da.where(ds[sig_name] > 0), lab, suf)
            else:
                print(f"[warn] Missing significance var {sig_name}; cannot mask {lab}.")

    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    for da, label, suf in to_plot:

        if auto_range:
            vmin, vmax = robust_sym_limits(da, q=q)
        else:
            # keep behaviour simple; you can swap to fixed vmin/vmax later if needed
            vmin, vmax = robust_sym_limits(da, q=q)

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

        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        cbar.set_label(f"{label} slope (per year)")

        base = f"{var_name}_ts_{label.lower()}"
        if sig_only:
            base += "_sig"

        outfile = os.path.join(out_dir, f"{base}.png")
        fig.savefig(outfile, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Standalone colorbar (SVG)
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
        fig_cb.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
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
    p.add_argument("--var", default=None,
                   help="Variable prefix (e.g., sm, Et, precip). Required only if --dataset_path is omitted.")
    p.add_argument("--suffix", default=None,
                   help="Optional suffix for inferred dataset (e.g., breakpoints_pettitt).")
    p.add_argument("--dataset_path", default=None,
                   help="Optional override path to *_ts.zarr (if omitted, inferred from config + --var + --suffix).")
    p.add_argument("--outdir", default=None,
                   help="Optional override output directory (if omitted, uses outputs_root/figures/theil_sen/<var>/).")

    p.add_argument("--sig_only", action="store_true", help="Mask non-significant slopes")
    p.add_argument("--auto_range", action=argparse.BooleanOptionalAction, default=True,
                   help="Auto symmetric color limits (robust). Default: True. Use --no-auto_range to disable.")
    p.add_argument("--q", type=float, default=0.995, help="Quantile for auto_range (default 0.995)")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))

    def _sfx(s):
        if not s:
            return ""
        s = str(s).strip()
        return s if s.startswith("_") else f"_{s}"

    if args.dataset_path:
        ds_path = Path(args.dataset_path)
    else:
        if not args.var:
            raise SystemExit("If --dataset_path is omitted, you must provide --var (e.g., sm, Et, precip).")
        ds_path = outputs_root / "zarr" / f"out_{args.var}{_sfx(args.suffix)}_ts.zarr"

    if not ds_path.exists():
        raise SystemExit(f"Dataset not found: {ds_path}")

    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        if not args.var:
            raise SystemExit("If --outdir is omitted, you must provide --var so we can infer an output folder.")
        out_dir = outputs_root / "figures" / "theil_sen" / args.var

    out_dir.mkdir(parents=True, exist_ok=True)

    ds = open_source_ds(str(ds_path))

    plot_theilsen(
        ds, args.var if args.var else args.var, str(out_dir),
        sig_only=args.sig_only,
        auto_range=args.auto_range,
        q=args.q
    )

    print("Done!")


if __name__ == "__main__":
    main()