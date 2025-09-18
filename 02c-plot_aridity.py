#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray  # for .rio

"""
Grouped aridity stacked bars (Decrease/Neutral/Increase) per indicator.

Example:
    python 02c-plot_aridity.py \
        --dataset /mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr \
        --var Et \
        --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2
        -- mode combined
    python 02c-plot_aridity.py \
        --dataset /mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr \
        --var sm \
        --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2
    python 02c-plot_aridity.py \
        --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr \
        --var precip \
        --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2
    
"""

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot aridity-class summary bars for indicators or combined resilience signals."
    )
    ap.add_argument("--dataset", required=True, help="Path to NetCDF/Zarr dataset (e.g., out_sm_kt.zarr)")
    ap.add_argument("--var", required=True, help="Variable prefix (e.g., sm, Et, precip)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["indicators", "combined"], default="indicators",
                    help="Plot raw indicators (5 panels) or composite signals (5 panels).")
    return ap.parse_args()

# ---------------- Indicators (as in your biome script) ----------------
INDICATORS = [
    ("ac1",  "AC1"),
    ("std",  "SD"),
    ("fd",   "FD"),
    ("skew", "Skew"),
    ("kurt", "Kurt"),
]

# ---------------- Aridity binning ----------------
ARIDITY_BINS = [0, 0.03, 0.2, 0.35, 0.5, 0.65, 0.8, 1, 1.25, 1.5, np.inf]
ARIDITY_LABELS = [
    "Hyper-arid",
    "Arid",
    "Semi-arid",
    "Dry sub-humid 1",
    "Dry sub-humid 2",
    "Dry sub-humid 3",
    "Moist sub-humid",
    "Humid 1",
    "Humid 2",
    "Humid 3",
]

""" ARIDITY_BINS = [0, 0.05, 0.20, 0.50, 0.65, np.inf] # comment this out for 10 bins, 5 is the typical ones 
ARIDITY_LABELS = [
    "Hyper-arid",
    "Arid",
    "Semi-arid",
    "Dry subhumid",
    "Humid",
] """


ARIDITY_PLOT = ARIDITY_LABELS 

# ---------------- Helpers ----------------
def wrap_to_180(ds, lon_name="lon"):
    ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
    return ds

def get_area_grid(lat, dlon=0.25, dlat=0.25):
    """Area (km²) per grid cell at each latitude for a rectilinear grid."""
    R = 6371.0  # km
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    # spherical quadrangle: R^2 * dlon * (sin(phi+ dphi/2) - sin(phi - dphi/2))
    # but for small dphi, cos(phi)*dphi approx is fine and matches your notebook
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)

def area_da_for_grid(lat, lon):
    area_per_lat = get_area_grid(lat)
    area_grid = np.broadcast_to(area_per_lat[:, np.newaxis], (lat.size, lon.size))
    return xr.DataArray(area_grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

def tri_split_areas(kt, pval, area_da):
    valid = kt.notnull()
    total = float(area_da.where(valid).sum().values) if valid.any() else 0.0
    if total == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    inc = (pval < 0.05) & (kt > 0)
    dec = (pval < 0.05) & (kt < 0)
    neu = valid & ~(inc | dec)
    inc_a = float(area_da.where(inc).sum().values)
    dec_a = float(area_da.where(dec).sum().values)
    neu_a = float(area_da.where(neu).sum().values)
    return dec_a, neu_a, inc_a, total

def compute_aridity_classes_like(ds_target):
    """Compute time-mean aridity index (precip/PET), bin to classes, and
    return a DataArray of labels on the ds_target grid (lat, lon)."""

    # --- Precip (ERA5 monthly total precip 2000–2023) ---
    precip = xr.open_dataset(
        "/mnt/data/romi/data/ERA5_0.25_monthly/total_precipitation/total_precipitation_monthly.nc"
    ).sel(time=slice("2000-01-01", "2023-12-31"))
    precip = precip.rename({"latitude": "lat", "longitude": "lon"})
    precip = precip.rio.write_crs("EPSG:4326")
    precip = wrap_to_180(precip, "lon")
    precip.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Land-sea mask to drop ocean
    ds_mask = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")
    ds_mask = ds_mask.rename({"latitude": "lat", "longitude": "lon"})
    ds_mask = ds_mask.rio.write_crs("EPSG:4326")
    ds_mask = wrap_to_180(ds_mask, "lon")
    ds_mask.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Interp mask to precip grid and apply
    mask_interp = ds_mask["lsm"].interp(lat=precip.lat, lon=precip.lon)
    precip = precip.where(mask_interp > 0.7)

    # m/month → mm/month
    precip = precip * 1000.0
    precip = precip.drop_vars(["number", "step", "surface", "spatial_ref"], errors="ignore")

    # Force time to month-start
    precip["time"] = ("time", pd.date_range(
        start=str(precip.time.values[0])[:10],
        periods=precip.sizes["time"],
        freq="MS"
    ))

    # --- PET (monthly) ---
    pet = xr.open_dataset("/mnt/data/romi/data/et_pot/monthly_sum_epot_clean.zarr").sel(
        time=slice("2000-01-01", "2023-11-30")
    )
    # Convert annual sum to monthly mean (from your snippet)
    pet["pet"] = pet["pet"] / 12.0

    pet["time"] = ("time", pd.date_range(
        start=str(pet.time.values[0])[:10],
        periods=pet.sizes["time"],
        freq="MS"
    ))

    # Avoid zeros
    precip_tp = precip["tp"].where(precip["tp"] != 0, 1e-6)
    pet_pet   = pet["pet"].where(pet["pet"] != 0, 1e-6)

    # Align by time
    precip_aligned, pet_aligned = xr.align(precip_tp, pet_pet, join="inner")

    # Aridity index and time-mean
    aridity_index = (precip_aligned / pet_aligned).rename("aridity_index")
    aridity_index = aridity_index.where(mask_interp > 0.7)
    aridity_mean = aridity_index.mean(dim="time")

    # Interp to analysis grid
    aridity_on_target = aridity_mean.interp(lat=ds_target.lat, lon=ds_target.lon)

    # Bin to classes (labels)
    class_index = xr.apply_ufunc(
        np.digitize,
        aridity_on_target,
        kwargs={"bins": ARIDITY_BINS},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    )

    # Map to labels 1..len(labels)
    def idx_to_label(idx):
        if 1 <= idx <= len(ARIDITY_LABELS):
            return ARIDITY_LABELS[idx - 1]
        return np.nan

    label_array = xr.apply_ufunc(
        np.vectorize(idx_to_label),
        class_index,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
    )

    label_array = label_array.rename("aridity_class")
    label_array = label_array.assign_coords(lat=ds_target.lat, lon=ds_target.lon)
    return label_array


def combined_signal_masks(ds, prefix):
    """
    Return dict[str -> xr.DataArray(bool)] for:
      - CSD: AC1↑ & SD↑
      - CSU: AC1↓ & SD↓
      - Mixed: (AC1↑ & SD↓) | (AC1↓ & SD↑)
      - FD↓: FD↓
      - Flickering: Skew↑ & Kurt↑
    All directions must be significant at p < 0.05.
    """
    ac1  = ds[f"{prefix}_ac1_kt"];  ac1_p  = ds[f"{prefix}_ac1_pval"]
    std  = ds[f"{prefix}_std_kt"];  std_p  = ds[f"{prefix}_std_pval"]
    fd   = ds[f"{prefix}_fd_kt"];   fd_p   = ds[f"{prefix}_fd_pval"]
    skew = ds[f"{prefix}_skew_kt"]; skew_p = ds[f"{prefix}_skew_pval"]
    kurt = ds[f"{prefix}_kurt_kt"]; kurt_p = ds[f"{prefix}_kurt_pval"]

    sig = lambda stat_p: (stat_p < 0.05)

    CSD  = (sig(ac1_p) & (ac1 > 0)) & (sig(std_p) & (std > 0))
    CSU  = (sig(ac1_p) & (ac1 < 0)) & (sig(std_p) & (std < 0))
    MIX1 = (sig(ac1_p) & (ac1 > 0)) & (sig(std_p) & (std < 0))
    MIX2 = (sig(ac1_p) & (ac1 < 0)) & (sig(std_p) & (std > 0))
    Mixed = (MIX1 | MIX2)

    FDdecline = (sig(fd_p) & (fd < 0))
    Flicker   = (sig(skew_p) & (skew > 0)) & (sig(kurt_p) & (kurt > 0))

    return {
        "CSD (AC1↑ & SD↑)": CSD,
        "CSU (AC1↓ & SD↓)": CSU,
        "Mixed (AC1↕ & SD↕)": Mixed,
        "FD↓": FDdecline,
        "Flickering (Skew↑ & Kurt↑)": Flicker,
    }


# ---------------- Main ----------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Kt output dataset (with _kt and _pval)
    ds = xr.open_dataset(args.dataset)

    # Urban mask 
    urban_mask = xr.open_dataset("/mnt/data/romi/data/urban_mask.zarr").rio.write_crs("EPSG:4326")
    urban_mask = urban_mask.interp_like(ds, method="nearest")
    urban_mask = urban_mask["urban-coverfraction"].squeeze("time")

    # land-sea mask to keep only land for precipitation
    # If the ds already excludes ocean it still works fine 
    lsm = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")
    lsm = lsm.rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = wrap_to_180(lsm, "lon")
    lsm = lsm.rio.set_spatial_dims("lon", "lat", inplace=True) or lsm
    lsm_interp = lsm["lsm"].interp(lat=ds.lat, lon=ds.lon)

    # Keep terrestrial, non-urban (≤3%) pixels
    terrestrial = (lsm_interp > 0.5) | lsm_interp.isnull()
    non_urban   = (urban_mask <= 3) | urban_mask.isnull()
    valid_mask  = terrestrial & non_urban

    # Compute aridity classes on ds grid
    aridity_class = compute_aridity_classes_like(ds).where(valid_mask)

    # Build containers: {indicator_key: {aridity_label: {dec,neu,inc,tot}}}
    group_totals = {
        k: {lab: {"dec": 0.0, "neu": 0.0, "inc": 0.0, "tot": 0.0} for lab in ARIDITY_PLOT}
        for k, _ in INDICATORS
    }

    # Pre-compute area on ds grid (assumes 0.25)
    lat = ds["lat"].values
    lon = ds["lon"].values
    dlat = float(np.abs(np.diff(lat).mean())) if lat.size > 1 else 0.25
    dlon = float(np.abs(np.diff(lon).mean())) if lon.size > 1 else 0.25

    # area function assumes 0.25 degrees resolution 
    def area_da_custom(lat_vals, lon_vals):
        R = 6371.0
        dlat_rad = np.radians(dlat)
        dlon_rad = np.radians(dlon)
        lat_rad = np.radians(lat_vals)
        area_row = (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
        grid = np.broadcast_to(area_row[:, None], (lat_vals.size, lon_vals.size))
        return xr.DataArray(grid, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))

    area_da = area_da_custom(lat, lon).where(valid_mask)

    if args.mode == "indicators":
        # Build area totals per class per indicator
        group_totals = {
            k: {lab: {"dec": 0.0, "neu": 0.0, "inc": 0.0, "tot": 0.0} for lab in ARIDITY_PLOT}
            for k, _ in INDICATORS
        }

        for label in ARIDITY_PLOT:
            cls_mask = (aridity_class == label)
            if not bool(cls_mask.any()):
                continue
            area_cls = area_da.where(cls_mask)

            for key, _lab in INDICATORS:
                kt   = ds[f"{args.var}_{key}_kt"].where(cls_mask)
                pval = ds[f"{args.var}_{key}_pval"].where(cls_mask)
                dec_a, neu_a, inc_a, tot_a = tri_split_areas(kt, pval, area_cls)
                acc = group_totals[key][label]
                acc["dec"] += dec_a; acc["neu"] += neu_a; acc["inc"] += inc_a; acc["tot"] += tot_a

        # Build DataFrames
        dfs = {}
        for key, _lab in INDICATORS:
            rows = []
            for lab in ARIDITY_PLOT:
                tot = group_totals[key][lab]["tot"]
                if tot > 0:
                    dec = 100.0 * group_totals[key][lab]["dec"] / tot
                    neu = 100.0 * group_totals[key][lab]["neu"] / tot
                    inc = 100.0 * group_totals[key][lab]["inc"] / tot
                    rows.append({"Aridity": lab, "Decrease": dec, "Neutral": neu, "Increase": inc})
            df = pd.DataFrame(rows) if rows else pd.DataFrame(
                {"Aridity": ARIDITY_PLOT, "Decrease": 0.0, "Neutral": 0.0, "Increase": 0.0}
            )
            df["Aridity"] = pd.Categorical(df["Aridity"], categories=ARIDITY_PLOT, ordered=True)
            dfs[key] = df.sort_values("Aridity")

        # Plot
        sn.set_style("whitegrid")
        fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
        colors = {"Decrease": "#64afc9ff", "Neutral": "#bab9b9", "Increase": "#cd5d5dff"}

        for ax, (key, label) in zip(axes, INDICATORS):
            dfp = dfs[key].copy()
            left = np.zeros(len(dfp))
            for col in ["Decrease", "Neutral", "Increase"]:
                ax.barh(y=dfp["Aridity"], width=dfp[col], left=left,
                        color=colors[col], edgecolor="none", height=0.87)
                left += dfp[col].to_numpy()
            ax.set_title(label, fontsize=14, pad=8)
            ax.set_xlim(0, 100); ax.set_xlabel("% of class area"); ax.grid(False)

        axes[0].invert_yaxis()
        for ax in axes:
            ax.tick_params(axis="y", labelsize=11)
            sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
            ax.margins(y=0.02)

        plt.tight_layout()
        out_svg = os.path.join(args.outdir, f"{args.var}_kt_aridity_indicators.svg")
        fig.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        plt.show(); plt.close(fig)
        return

    # -------- Mode: combined (5 composite panels) --------
    combos = combined_signal_masks(ds, args.var)

    # For each aridity class and combo, compute % of class area where combo True
    # Prepare dataframe per combo
    dfs = {}
    for cname, cmask in combos.items():
        rows = []
        for lab in ARIDITY_PLOT:
            cls_mask = (aridity_class == lab)
            if not bool(cls_mask.any()):
                continue
            area_cls = area_da.where(cls_mask)
            tot = float(area_cls.sum().values)
            if tot == 0.0:
                pct = 0.0
            else:
                hit_area = float(area_cls.where(cmask & cls_mask).sum().values)
                pct = 100.0 * hit_area / tot
            rows.append({"Aridity": lab, "Percent": pct})
        df = pd.DataFrame(rows) if rows else pd.DataFrame({"Aridity": ARIDITY_PLOT, "Percent": 0.0})
        df["Aridity"] = pd.Categorical(df["Aridity"], categories=ARIDITY_PLOT, ordered=True)
        dfs[cname] = df.sort_values("Aridity")

    # Plot: one horizontal bar panel per composite
    sn.set_style("whitegrid")
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for ax, (cname, dfp) in zip(axes, dfs.items()):
        ax.barh(y=dfp["Aridity"], width=dfp["Percent"], edgecolor="none", height=0.87)
        ax.set_title(cname, fontsize=13, pad=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of class area")
        ax.grid(False)

    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="y", labelsize=11)
        sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
        ax.margins(y=0.02)

    plt.tight_layout()
    out_svg = os.path.join(args.outdir, f"{args.var}_kt_aridity_combined.svg")
    fig.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close(fig)

if __name__ == "__main__":

    plt.rc("figure", figsize=(13, 8))
    plt.rc("font", size=12)
    sn.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
 
    main()