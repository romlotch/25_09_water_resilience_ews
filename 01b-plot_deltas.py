
import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sn

import warnings 
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from utils.config import load_config, cfg_path, cfg_get

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader



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

    if var_name.lower() == "sm":
        vmins = [-1.5, -0.075, -10, -35, -0.5]
        vmaxs = [ 1.5,  0.075,  10,  35,  0.5]
    elif var_name.lower() == "et":
        vmins = [-1.5, -1, -10, -35, -1]
        vmaxs = [ 1.5,  1,  10,  35,  1]
    else:  # precip
        vmins = [-1, -30, -10, -35, -0.5]
        vmaxs = [ 1,  30,  10,  35,  0.5]

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
        outfile = Path(outdir) / f"{var_name}_deltas_{label.lower()}.png"
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
        colorbar_outfile = Path(outdir) / f"colorbar_{var_name}_deltas_{label.lower()}.svg"
        plt.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f'Saved to: {outfile}')

def main(): 

    warnings.filterwarnings("ignore")

    sn.set_style("white")
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "DejaVu Sans"

    p = argparse.ArgumentParser()
    p.add_argument("--variable", required=True, help="sm | Et | precip")
    p.add_argument("--suffix", default=None, help="Optional suffix used in filename (e.g. 'breakpoint_stc').")
    p.add_argument("--dataset_path", default=None, help="Optional override input Zarr path.")
    p.add_argument("--output_path", default=None, help="Optional override output directory.")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = p.parse_args()

    cfg = load_config(args.config)
    var_name = args.variable

    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)

    # Default input: outputs/zarr/out_<var><_suffix>.zarr
    sfx = "" if not args.suffix else (args.suffix if args.suffix.startswith("_") else f"_{args.suffix}")
    default_in = Path(outputs_root) / "zarr" / f"out_{var_name}{sfx}.zarr"
    ds_path = Path(args.dataset_path) if args.dataset_path else default_in

    # Default output: outputs/figures/deltas/<var>/
    default_out = Path(outputs_root) / "figures" / "deltas" / var_name
    out_dir = Path(args.output_path) if args.output_path else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01b] Reading: {ds_path}")
    ds = xr.open_dataset(ds_path)


    if var_name == "precip":
        mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
        ds_mask = xr.open_dataset(mask_path, engine="cfgrib")

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

    
    plot_deltas(ds, var_name, out_path=str(out_dir) + os.sep)

    print('Done!')


if __name__=='__main__':

    main()
