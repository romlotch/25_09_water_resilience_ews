import os
import argparse
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

""" 
Plots examples of raw variable time series with abrupt shifts marked. 

Max Fstat needs to be adjusted depending on variable. Currently selects the pixels 
with the highest Fstat based on Fmax. 

Path to land mask is hardcoded. 

Inputs: 
    --raw: path to original ews output with raw series
    --chp: path to chp output
    --var: variable 
    --outdir: where to save plots to 

E.g. 
    python 03b-plot_example_abrupt_shift.py --raw '/mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr' --chp '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr' --var 'sm' --outdir '/mnt/data/romi/figures/paper_1/supplementary_final/supp_abrupt_shifts'
    python 03b-plot_example_abrupt_shift.py --raw '/mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr' --chp '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr' --var 'Et' --outdir '/mnt/data/romi/figures/paper_1/supplementary_final/supp_abrupt_shifts'
    python 03b-plot_example_abrupt_shift.py --raw '/mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr' --chp '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr' --var 'precip' --outdir '/mnt/data/romi/figures/paper_1/supplementary_final/supp_abrupt_shifts'

"""

# ---helpers ---
def open_source_ds(path):
    """Open NetCDF or Zarr by path."""
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zmetadata"))
    ):
        return xr.open_zarr(path)
    return xr.open_dataset(path)

def mask_precip_to_land(ds_p):
    """Return precip datasetwith land mask"""

    LANDSEA_PATH = "/mnt/data/romi/data/landsea_mask.grib"
    
    ds_mask = xr.open_dataset(LANDSEA_PATH, engine="cfgrib")
    ds_mask = ds_mask.rio.write_crs("EPSG:4326")
    
    ds_mask = ds_mask.assign_coords(
        longitude=(((ds_mask.longitude + 180) % 360) - 180)
    ).sortby("longitude")
    ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)

    # interpolate mask to precip grid
    ds_mask_i = ds_mask.interp(
        longitude=ds_p.lon, latitude=ds_p.lat, method="nearest"
    )

    # land mask, rename dims to match precip
    land = (ds_mask_i["lsm"] > 0.7)

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

    land = land.reindex_like(ds_p, method=None)
    
    return ds_p.where(land)

def cp_index_to_datetime(time_da, cp_index):
    """
    Convert a changepoint index to a datetime .
    Infer time step from median delta of the time coordinate.
    """
    if time_da.size == 0 or np.isnan(cp_index):
        return np.datetime64('NaT')
    t0 = np.array(time_da.values[0], dtype='datetime64[ns]')
    if time_da.size > 1:
        step = np.median(np.diff(time_da.values).astype('timedelta64[ns]'))
    else:
        step = np.timedelta64(7, 'D').astype('timedelta64[ns]') # this doesnt happen
    offset = (np.float64(cp_index) * np.int64(step)).astype('timedelta64[ns]')
    return (t0 + offset).astype('datetime64[ns]')

def pick_example_pixels(ds_chp, n, fmax):
    """
    Select N pixels with highest F-stats as examples, but cant be more
    than specified fmax. 
    """
    needed = ["Fstat", "Fstat_pval", "strucchange_bp"]
    for v in needed:
        if v not in ds_chp:
            raise KeyError(f"Missing '{v}' in changepoint dataset.")

    F     = ds_chp["Fstat"]
    pval  = ds_chp["Fstat_pval"]
    bp    = ds_chp["strucchange_bp"]

    # significance + valid bp
    base_mask = (pval < 0.05) & (bp > 0) & np.isfinite(F)

    # cap by Fstat
    mask = base_mask & (F <= fmax)

    Fsig = F.where(mask)
    Fvals = Fsig.values
    idx = np.argwhere(np.isfinite(Fvals))

    # if theres nothing under fmax just use fmax so it doesnt error
    if idx.size == 0:
        Fsig = F.where(base_mask)
        Fvals = Fsig.values
        idx = np.argwhere(np.isfinite(Fvals))
        if idx.size == 0:
            return []

    scores = Fvals[idx[:, 0], idx[:, 1]]
    order = np.argsort(scores)[::-1]   # descending
    chosen = idx[order][:n]
    return [(int(i), int(j)) for i, j in chosen]


def set_ylim_mean_centered(ax, y, pad=0.05, draw_mean=False):
    """
    Centre ylims on mean
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        ax.set_ylim(0, 1.0)
        return

    y_mean = float(y.mean())
    y_max  = float(y.max())
    ymax   = max(2.0 * y_mean, (1.0 + pad) * y_max)
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    ax.set_ylim(0.0, ymax)
    if draw_mean:
        ax.axhline(y_mean, ls="--", lw=0.8, alpha=0.5, color="k")

# --- plot ---
def plot_example_timeseries(raw_ds, chp_ds, var_name,  fmax, outdir, n_examples = 12):
    
    if var_name.lower() in ("precip", "p", "pr"):
        raw_ds = mask_precip_to_land(raw_ds)
        chp_ds = mask_precip_to_land(chp_ds)
        
    # pick pixels
    picks = pick_example_pixels(chp_ds, n_examples, fmax = fmax)
    if not picks:
        raise RuntimeError("No significant abrupt shifts found (Fstat test).")

    n = len(picks)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False, sharex=False, sharey=False)

    time = raw_ds["time"]
    da   = raw_ds[var_name]

    # cp vars
    F     = chp_ds["Fstat"]
    pval  = chp_ds["Fstat_pval"]
    bp    = chp_ds["strucchange_bp"]

    # Plot each example
    for k, (ilat, ilon) in enumerate(picks):
        r = k // ncols
        c = k % ncols
        ax = axes[r][c]

        # extract series
        ts = da.isel(lat=ilat, lon=ilon)
        x = time.values
        y = ts.values

        # breakpoint =  datetime on same axis
        cp_val = float(bp.isel(lat=ilat, lon=ilon).values)
        x_cp   = cp_index_to_datetime(time, cp_val)

        # plot
        ax.plot(x, y, lw=0.5, color="#333333")
        if np.isfinite(cp_val) and not np.isnat(x_cp):
            ax.axvline(x_cp, color="#cd5d5d", lw=1.5, alpha=0.9)

        # title with lat/lon and stats
        latv = float(raw_ds["lat"].values[ilat])
        lonv = float(raw_ds["lon"].values[ilon])
        fval = float(F.isel(lat=ilat, lon=ilon).values)
        pvalv = float(pval.isel(lat=ilat, lon=ilon).values)
        ax.set_title(f"Latitude: {latv:.2f}, Longitude: {lonv:.2f}", fontsize=9)

        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name in ("left", "bottom"))
            if spine.get_visible():
                spine.set_linewidth(0.8)
                spine.set_color("black")
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8)

        # set_ylim_mean_centered(ax, ts.values, pad=0.05, draw_mean=False)

    # turn off unused axes
    for k in range(n, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r][c].axis("off")

    fig.suptitle(f"", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(outdir, exist_ok=True)
    svg = os.path.join(outdir, f"{var_name}_example_timeseries_stc.svg")
    fig.savefig(svg, format = 'svg', dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {svg}")

# --- main ---
def main():

    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'

    p = argparse.ArgumentParser(description="Plot example raw time series at pixels with detected abrupt shifts (StructChange/F).")
    p.add_argument("--raw",  required=True, help="Path to raw dataset (e.g., .../out_sm.zarr)")
    p.add_argument("--chp",  required=True, help="Path to changepoint dataset (e.g., .../out_sm_chp.zarr)")
    p.add_argument("--var",  required=True, choices=["sm", "Et", "precip"], help="Variable name in the raw dataset")
    p.add_argument("--outdir", required=True, help="Output directory for the figure")
    p.add_argument("--n", type=int, default=12, help="How many examples to plot (default 12)")
    args = p.parse_args()

    raw_ds = open_source_ds(args.raw)
    chp_ds = open_source_ds(args.chp)

    if args.var == 'sm': 

        plot_example_timeseries(raw_ds, chp_ds, args.var, fmax = 150, n_examples=3, outdir = args.outdir)

    elif args.var == 'Et': 

        plot_example_timeseries(raw_ds, chp_ds, args.var, fmax = 19, n_examples=3, outdir = args.outdir)

    elif args.var == 'precip':

        plot_example_timeseries(raw_ds, chp_ds, args.var, fmax = 100, n_examples=3, outdir = args.outdir)


    
if __name__ == "__main__":

    main()