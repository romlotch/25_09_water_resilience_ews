#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

"""
Per-aridity stacked bar plots (TP/FP/FN/TN) for combined EWS indicators,
using combined pre-break (positives) and pseudo-break (negatives) Kendall tau datasets.

Combine the EWS mask as:
  e_all = e_pos  where breakpoint==True   (from pre-break tau dataset)
          e_neg  where breakpoint==False  (from pseudo-break tau dataset)


Inputs:
  --cp_path       : changepoint dataset with fields for three methods
                    (pettitt_cp,pettitt_pval), (strucchange_bp,Fstat_pval), (bp_var,pval_var)
  --pre_pos_path  : pre-break tau dataset (computed before the real CP; “positives”)
  --pre_neg_path  : pre-pseudo-break tau dataset (computed before sampled pseudo CP; “negatives”)
  --var           : variable prefix in tau datasets (e.g., sm, Et, precip)
  --out_dir       : where to save figures/CSVs
  --alpha         : significance level for tau p-values (default 0.05)

Example:
  python 05a-plot_true_positives_pre.py \
    --cp_path /home/romi/ews/output/paper_1/output_sm_final/out_sm_chp.zarr \
    --pre_pos_path /home/romi/ews/output/paper_1/output_sm_final/out_sm_breakpoint_stc_kt.zarr \
    --pre_neg_path /home/romi/ews/output/paper_1/output_sm_final/out_sm_breakpoint_stc_neg_kt.zarr \
    --var sm \
    --out_dir /home/romi/ews/figures/figure_4

  python 05a-plot_true_positives_pre.py \
    --cp_path /home/romi/ews/output/output_Et_final/out_Et_chp.zarr \
    --pre_pos_path /home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc_kt.zarr \
    --pre_neg_path /home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc_neg_kt.zarr \
    --var Et \
    --out_dir /home/romi/ews/figures/figure_4


"""

# ---- Config ---
ARIDITY_BINS   = [0, 0.05, 0.20, 0.50, 0.65, np.inf]
ARIDITY_LABELS = ["Hyper-arid", "Arid", "Semi-arid", "Dry subhumid", "Humid"]
COMPOSITES = ['AC1 ↑ & SD ↑', 'AC1 ↓ & SD ↓', 'AC1 ↑↓ & SD ↑↓', 'Skew ↑ & Kurt ↑', 'FD ↓']
CLASS_COLORS = {'TP': '#7d475dff', 'FP': '#ffeaefff', 'FN': '#e9e2bcff', 'TN': '#2c787eff'}
CP_METHODS = {
    'pettitt':  ('pettitt_cp',     'pettitt_pval'),
    'stc':      ('strucchange_bp', 'Fstat_pval'),
    'variance': ('bp_var',         'pval_var'),
}

# --- Helpers ---
def detect_lon_lat_names(ds):
    lon = next((c for c in ["lon","longitude","x"] if c in ds.coords), None)
    lat = next((c for c in ["lat","latitude","y"] if c in ds.coords), None)
    if lon is None or lat is None:
        raise ValueError(f"Could not find lon/lat in coords: {list(ds.coords)}")
    return lon, lat

def wrap_to_180(da, lon_name="lon"):
    return da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)}).sortby(lon_name)

def get_area_da(lat_vals, lon_vals):
    """Area (km2) per grid cell (spherical quad)."""
    R = 6371.0
    dlat = float(np.abs(np.diff(lat_vals).mean())) if len(lat_vals) > 1 else 0.25
    dlon = float(np.abs(np.diff(lon_vals).mean())) if len(lon_vals) > 1 else 0.25
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    lat_rad  = np.radians(lat_vals)
    row = (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
    grid = np.broadcast_to(row[:, None], (lat_vals.size, lon_vals.size))
    return xr.DataArray(grid, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat","lon"))

def compute_aridity_classes_like(ds_target):
    """
    Compute time-mean aridity index (precip/PET) and bin to ARIDITY_LABELS on ds_target grid.
   
    """
    AI_MEAN_MAX = 100.0        

    def wrap_to_180(ds, lon_name="lon"):
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
        return ds

    # --- Precip (ERA5 monthly total precip 2000–2023) ---
    precip = xr.open_dataset(
        "/home/romi/ews/data/total_precipitation_monthly.nc"
    ).sel(time=slice("2000-01-01", "2023-12-31"))
    precip = precip.rename({"latitude": "lat", "longitude": "lon"})
    precip = precip.rio.write_crs("EPSG:4326")
    precip = wrap_to_180(precip, "lon")
    precip.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Land-sea mask to drop ocean
    ds_mask = xr.open_dataset("/home/romi/ews/data/landsea_mask.grib")
    ds_mask = ds_mask.rename({"latitude": "lat", "longitude": "lon"})
    ds_mask = ds_mask.rio.write_crs("EPSG:4326")
    ds_mask = wrap_to_180(ds_mask, "lon")
    ds_mask.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Interp mask to precip grid and apply
    mask_interp = ds_mask["lsm"].interp(lat=precip.lat, lon=precip.lon)
    precip = precip.where(mask_interp > 0.7)

    # m/month to mm/month
    precip = precip * 1000.0
    precip = precip.drop_vars(["number", "step", "surface", "spatial_ref"], errors="ignore")

    # Force time to month-start
    precip["time"] = ("time", pd.date_range(
        start=str(precip.time.values[0])[:10],
        periods=precip.sizes["time"],
        freq="MS"
    ))

    # --- PET (monthly) ---
    pet = xr.open_dataset("/home/romi/ews/data/monthly_sum_epot_clean.zarr").sel(
        time=slice("2000-01-01", "2023-11-30")
    )
    pet["time"] = ("time", pd.date_range(
        start=str(pet.time.values[0])[:10],
        periods=pet.sizes["time"],
        freq="MS"
    ))

    # Avoid zeros (numerical guard)
    precip_tp = precip["tp"].where(precip["tp"] != 0, 1e-6)
    pet_pet   = pet["pet"].where(pet["pet"] != 0, 1e-6)

    # Align by time
    precip_aligned, pet_aligned = xr.align(precip_tp, pet_pet, join="inner")

    # AI = ratio of annual sums 
    P_y   = precip_aligned.resample(time="YS").sum()
    PET_y = pet_aligned.resample(time="YS").sum().where(lambda x: x != 0, 1e-6)

    ai_annual = (P_y / PET_y).rename("aridity_index")
    ai_annual = ai_annual.where(mask_interp > 0.7)

    # Mask absurd annual values before computing stats
    ai_annual = ai_annual.where(ai_annual <= AI_MEAN_MAX)
    ai_mean = ai_annual.mean("time")

    lon, lat = detect_lon_lat_names(ds_target)
    ai_on_target = ai_mean.interp(lat=ds_target[lat], lon=ds_target[lon])

    idx = xr.apply_ufunc(np.digitize, ai_on_target, kwargs={"bins": ARIDITY_BINS},
                         vectorize=True, dask="parallelized", output_dtypes=[int])

    def idx2label(i):
        return ARIDITY_LABELS[i-1] if (1 <= i <= len(ARIDITY_LABELS)) else np.nan

    labels = xr.apply_ufunc(np.vectorize(idx2label), idx,
                            vectorize=True, dask="parallelized", output_dtypes=[object]).rename("aridity_class")

    # Ensure lat/lon names match
    if lon != "lon" or lat != "lat":
        labels = labels.rename({lon:"lon", lat:"lat"})
    return labels

def availability_mask(ds, label, var):
    """Require tau and pval to exist """

    def have(v): return ds[f"{v}_kt"].notnull() & ds[f"{v}_pval"].notnull()
    if label == 'FD down':                return have(f"{var}_fd")
    if label == 'Skew up & Kurt up':     return have(f"{var}_skew") & have(f"{var}_kurt")
    
    return have(f"{var}_ac1") & have(f"{var}_std")

def ews_mask_from(ds, label, var, alpha=0.05):
   
    def dir_mask(vname, inc=True):
        tau = ds[f"{vname}_kt"]; p = ds[f"{vname}_pval"]
        return ((p < alpha) & ((tau > 0) if inc else (tau < 0))).fillna(False)

    if label == 'FD down':
        tau = ds[f"{var}_fd_kt"]; p = ds[f"{var}_fd_pval"]
        return ((p < alpha) & (tau < 0)).fillna(False)
    if label == 'Skew up & Kurt up':
        return dir_mask(f"{var}_skew", True) & dir_mask(f"{var}_kurt", True)
    if label == 'AC1 up & SD up':
        return dir_mask(f"{var}_ac1", True) & dir_mask(f"{var}_std", True)
    if label == 'AC1 down & SD down':
        return dir_mask(f"{var}_ac1", False) & dir_mask(f"{var}_std", False)
    if label == 'AC1 updown & SD updown':
        return (dir_mask(f"{var}_ac1", True) & dir_mask(f"{var}_std", False)) | \
               (dir_mask(f"{var}_ac1", False) & dir_mask(f"{var}_std", True))
    raise ValueError(f"Unknown composite: {label}")

def break_mask_from(ds_cp, method, alpha=0.05):
    cp_name, pv_name = CP_METHODS[method]
    cp = ds_cp[cp_name]; pv = ds_cp[pv_name]
    return ((pv < alpha) & (cp > 0)).fillna(False)

def compute_confusion_arrays(ews_bool, bp_bool):
    e = ews_bool.astype(bool)
    b = bp_bool.astype(bool)
    TP = e & b
    FP = e & ~b
    FN = ~e & b
    TN = ~e & ~b
    return TP, FP, FN, TN

# --- build long table & plot ---
def aridity_confusion_long(ds_cp, ds_pos, ds_neg, var, alpha=0.05):
    """
    Return long DF with area-weighted fractions per method, aridity, EWS,
    where the EWS mask is made from ds_pos (on breaks) and ds_neg (on no-breaks).
    """
    # interp
    template = ds_pos
    ds_neg   = ds_neg.interp_like(template, method="nearest")
    ds_cp    = ds_cp.interp_like(template, method="nearest")

    # Build classess and area weights
    aridity = compute_aridity_classes_like(template)
    A       = get_area_da(template["lat"].values, template["lon"].values)

    rows = []
    for method in CP_METHODS:
        bmask = break_mask_from(ds_cp, method, alpha=alpha)

        for comp in COMPOSITES:
            # check to use pos or neg
            avail_pos = availability_mask(ds_pos, comp, var)
            avail_neg = availability_mask(ds_neg, comp, var)

        
            eval_mask = (bmask & avail_pos) | ((~bmask) & avail_neg)

            # EWS masks on each dataset
            e_pos = ews_mask_from(ds_pos, comp, var, alpha).where(avail_pos, False)
            e_neg = ews_mask_from(ds_neg, comp, var, alpha).where(avail_neg, False)

            e_all = xr.where(bmask, e_pos, e_neg).where(eval_mask, False)
            b_all = bmask.where(eval_mask, False)

            # per aridity class area-weighted fractions
            for cls in ARIDITY_LABELS:
                cls_mask = (aridity == cls)
                if not bool(cls_mask.any()):
                    continue
                A_cls = A.where(cls_mask & eval_mask)

                tot = float(A_cls.sum(skipna=True).values)
                if tot <= 0:
                    continue

                TP, FP, FN, TN = compute_confusion_arrays(e_all & cls_mask, b_all & cls_mask)
                TP_a = float(A_cls.where(TP).sum(skipna=True).values)
                FP_a = float(A_cls.where(FP).sum(skipna=True).values)
                FN_a = float(A_cls.where(FN).sum(skipna=True).values)
                TN_a = float(A_cls.where(TN).sum(skipna=True).values)
                denom = TP_a + FP_a + FN_a + TN_a
                if denom <= 0:
                    continue

                rows += [
                    {"method": method, "aridity_class": cls, "label": comp, "cls": "TP", "frac": TP_a/denom, "area_km2": TP_a},
                    {"method": method, "aridity_class": cls, "label": comp, "cls": "FP", "frac": FP_a/denom, "area_km2": FP_a},
                    {"method": method, "aridity_class": cls, "label": comp, "cls": "FN", "frac": FN_a/denom, "area_km2": FN_a},
                    {"method": method, "aridity_class": cls, "label": comp, "cls": "TN", "frac": TN_a/denom, "area_km2": TN_a},
                ]

    df = pd.DataFrame(rows)
    if not df.empty:
        df["aridity_class"] = pd.Categorical(df["aridity_class"], categories=ARIDITY_LABELS, ordered=True)
    return df.sort_values(["method","label","aridity_class","cls"])



def plot_aridity_confusion_rows(df_long, var, out_dir):
    """
    Stacked bars per CP method (pettitt, stc, variance),
    for aridity classes for each EWS.
    """
    colors = CLASS_COLORS
    classes = ['TP','FP','FN','TN']
    methods = list(CP_METHODS.keys())

    for method in methods:
        df_m = df_long[df_long["method"] == method]
        if df_m.empty:
            continue

        labels_order = COMPOSITES
        titles       = COMPOSITES
        fig_tag      = "ews"

        sn.set_style("whitegrid")
        fig, axes = plt.subplots(1, 5, figsize=(3.9*5, 3.6), sharey=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, label, title in zip(axes, labels_order, titles):
            df_ml = df_m[df_m["label"] == label]
            if df_ml.empty:
                ax.set_title(title, pad=6)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Fraction of area')
                ax.set_yticklabels([])
                sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
                continue

            pivot = (df_ml
                     .pivot_table(index='aridity_class', columns='cls', values='frac', aggfunc='mean')
                     .reindex(ARIDITY_LABELS))
            pivot = pivot.fillna(0.0)

            left = np.zeros(len(pivot))
            for cls in classes:
                vals = pivot[cls].values if cls in pivot.columns else np.zeros(len(pivot))
                ax.barh(pivot.index, vals, left=left, label=cls, color=colors[cls], height=0.86)
                left += vals

            ax.set_xlim(0, 1)
            ax.set_title(title, pad=6)
            ax.set_xlabel('Fraction of area')
            if label == labels_order[0]:
                ax.set_yticklabels(pivot.index)
            else:
                ax.set_yticklabels([])
            ax.grid(False)
            sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)

        handles = [mpl.patches.Patch(color=colors[c], label=c) for c in classes]
        fig.legend(handles=handles, ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(f"{var}: Aridity confusion — {fig_tag} — {method}", y=1.18, fontsize=13)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"{var}_{method}_aridity_confusion_{fig_tag}_row.svg")
        fig.savefig(fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)



def main():
    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    parser = argparse.ArgumentParser(
        description="Per-aridity stacked bars for combined EWS using pre-break and pseudo-break τ datasets."
    )
    parser.add_argument('--cp_path',       type=str, required=True, help='Changepoint dataset.')
    parser.add_argument('--pre_pos_path',  type=str, required=True, help='Pre-break τ dataset (positives).')
    parser.add_argument('--pre_neg_path',  type=str, required=True, help='Pre-pseudo-break τ dataset (negatives).')
    parser.add_argument('--var',           type=str, required=True, help='Variable prefix (e.g., sm, Et, precip).')
    parser.add_argument('--out_dir',       type=str, required=True, help='Output directory.')
    parser.add_argument('--alpha',         type=float, default=0.05, help='Significance threshold for τ.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    ds_cp  = xr.open_dataset(args.cp_path)
    ds_pos = xr.open_dataset(args.pre_pos_path)
    ds_neg = xr.open_dataset(args.pre_neg_path)

    # Build long table
    df_long = aridity_confusion_long(ds_cp, ds_pos, ds_neg, var=args.var, alpha=args.alpha)

    # Save the numbers
    csv_fp = os.path.join(args.out_dir, f"{args.var}_aridity_confusion_ews_long.csv")
    df_long.to_csv(csv_fp, index=False)
    print(f"[save] {csv_fp}")

    # Plot
    plot_aridity_confusion_rows(df_long, args.var, args.out_dir)

    print("[done]")

if __name__ == "__main__":
    main()