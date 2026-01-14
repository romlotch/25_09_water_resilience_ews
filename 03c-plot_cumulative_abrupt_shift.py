#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.ticker as mticker
import matplotlib as mpl
from utils.config import load_config, cfg_path, cfg_get
from pathlib import Path

""" 
Plots and save cumulative land area with abrupt shifts for 
precipitation, transpiration, and soil moisture. 

E.g. 
    python3 03c-plot_cumulative_abrupt_shift.py --sm_path "/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr" --et_path "/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr" --p_path "/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr" --test 'stc' --outdir '/mnt/data/romi/figures/paper_1/supplementary_final/supp_abrupt_shifts'

"""

# --- defaults ---
SM_PATH = "/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr"
ET_PATH = "/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr"
P_PATH  = "/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr"


START_YEAR = 2000
WEEKS_PER_YEAR = 52.1775

# --- label mapping ---
CP_PVAL_MAP = {
    "pettitt": {"cp": "pettitt_cp",     "pval": "pettitt_pval", "label": "Pettitt"},
    "stc":     {"cp": "strucchange_bp", "pval": "Fstat_pval",   "label": "StructChange F"},
    "var":     {"cp": "bp_var",         "pval": "pval_var",     "label": "Variance break"},
}

def lat_band_area_km2(lat_deg, dlon=0.25, dlat=0.25):
    """1D area per latitude (km^2) for a regular lon/lat grid."""
    R = 6371.0  # km
    lat_rad = np.radians(lat_deg)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)


def mask_precip_to_land(ds_p: xr.Dataset, cfg) -> xr.Dataset:
    """Mask precip dataset to land (lsm>0.7) on ds_p's lon/lat grid."""
    mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
    ds_mask = xr.open_dataset(mask_path, engine="cfgrib")
    ds_mask = ds_mask.rio.write_crs("EPSG:4326")

    ds_mask = ds_mask.assign_coords(
        longitude=(((ds_mask.longitude + 180) % 360) - 180)
    ).sortby("longitude")
    ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)

    ds_mask_i = ds_mask.interp(
        longitude=ds_p.lon, latitude=ds_p.lat, method="nearest"
    )

    land = (ds_mask_i["lsm"] > 0.7)

    if "longitude" in land.dims:
        land = land.rename({"longitude": "lon"})
    if "latitude" in land.dims:
        land = land.rename({"latitude": "lat"})

    land = land.reindex_like(ds_p, method=None)
    return ds_p.where(land)


def compute_accumulated_percent(
    ds: xr.Dataset,
    cp_name: str,
    pval_name: str,
    start_year: int = START_YEAR,
    weeks_per_year: float = WEEKS_PER_YEAR,
):
    if cp_name not in ds or pval_name not in ds:
        raise KeyError(f"Dataset missing variables: {cp_name} or {pval_name}")

    cp = ds[cp_name]
    pval = ds[pval_name]

    sig = (pval < 0.05) & (cp > 0)
    if "lat" not in cp.dims or "lon" not in cp.dims:
        raise ValueError("Expected variables to have 'lat' and 'lon' dimensions.")

    lat = ds["lat"].values
    lon = ds["lon"].values
    dlon = float(abs(np.median(np.diff(lon))) if lon.size > 1 else 0.25)
    dlat = float(abs(np.median(np.diff(lat))) if lat.size > 1 else 0.25)

    area1d = lat_band_area_km2(lat, dlon=dlon, dlat=dlat)
    area2d = np.broadcast_to(area1d[:, None], (lat.size, lon.size))
    area_da = xr.DataArray(area2d, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    valid = cp.notnull()
    total_area = float(area_da.where(valid).sum().values)
    if total_area <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    cp_year_float = start_year + (cp / weeks_per_year)
    cp_year_int = np.floor(cp_year_float).astype("float64")
    cp_year_int = cp_year_int.where(sig)

    max_year = int(np.nanmax(cp_year_int.values)) if np.isfinite(cp_year_int.values).any() else start_year
    years = np.arange(start_year, max_year + 1, dtype=int)

    cumulative_area = np.zeros_like(years, dtype=float)
    cp_year_arr = cp_year_int.values
    area_arr = area_da.values

    for i, y in enumerate(years):
        mask_y = np.isfinite(cp_year_arr) & (cp_year_arr == float(y))
        yearly_area = area_arr[mask_y].sum()
        cumulative_area[i] = yearly_area if i == 0 else cumulative_area[i - 1] + yearly_area

    cumulative_percent = 100.0 * cumulative_area / total_area
    return years, cumulative_percent


def extend_series_to(years_base, years, vals):
    """Extend a shorter series to years_base via forward fill."""
    out = np.full_like(years_base, np.nan, dtype=float)
    year_to_val = {int(y): float(v) for y, v in zip(years, vals)}
    last = 0.0
    for i, y in enumerate(years_base):
        if int(y) in year_to_val:
            last = year_to_val[int(y)]
        out[i] = last
    return out


def plot_accumulated_three(sm_ds_path, et_ds_path, p_ds_path, test_name, outdir, cfg):
    if test_name not in CP_PVAL_MAP:
        raise ValueError(f"--test must be one of {list(CP_PVAL_MAP.keys())}")

    cp_name = CP_PVAL_MAP[test_name]["cp"]
    pval_name = CP_PVAL_MAP[test_name]["pval"]

    os.makedirs(outdir, exist_ok=True)

    ds_sm = xr.open_dataset(sm_ds_path)
    ds_et = xr.open_dataset(et_ds_path)
    ds_p = xr.open_dataset(p_ds_path)

    ds_p = mask_precip_to_land(ds_p, cfg)

    y_sm, s_sm = compute_accumulated_percent(ds_sm, cp_name, pval_name)
    y_et, s_et = compute_accumulated_percent(ds_et, cp_name, pval_name)
    y_p,  s_p  = compute_accumulated_percent(ds_p,  cp_name, pval_name)

    max_year = max([y[-1] if y.size else START_YEAR for y in (y_sm, y_et, y_p)])
    years = np.arange(START_YEAR, max_year + 1, dtype=int)

    s_sm_e = extend_series_to(years, y_sm, s_sm) if y_sm.size else np.zeros_like(years, dtype=float)
    s_et_e = extend_series_to(years, y_et, s_et) if y_et.size else np.zeros_like(years, dtype=float)
    s_p_e  = extend_series_to(years, y_p,  s_p)  if y_p.size  else np.zeros_like(years, dtype=float)

    pd.Series(s_sm_e, index=years, name="SM").to_csv(os.path.join(outdir, f"values_sm_{test_name}.csv"))
    pd.Series(s_et_e, index=years, name="ET").to_csv(os.path.join(outdir, f"values_et_{test_name}.csv"))
    pd.Series(s_p_e,  index=years, name="P").to_csv(os.path.join(outdir, f"values_p_{test_name}.csv"))

    sn.set_style("whitegrid")
    plt.figure(figsize=(7, 3))
    plt.plot(years, s_sm_e, label="Soil moisture", linewidth=2, color="#a58746")
    plt.plot(years, s_et_e, label="Transpiration", linewidth=2, color="#845B7D")
    plt.plot(years, s_p_e,  label="Precipitation", linewidth=2, color="#5FA7C0")

    plt.xlabel("Year")
    plt.ylabel("Cumulative land area [%]")
    plt.title("")
    plt.xlim(2002, 2020)
    plt.ylim(0, 50)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        if side in ("left", "bottom"):
            spine.set_linewidth(1.5)
            spine.set_color("black")

    svg = os.path.join(outdir, f"accumulated_cp_area_{test_name}.svg")
    plt.savefig(svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Time series of accumulated land area with abrupt shifts for SM/ET/Precip.")
    p.add_argument("--test", choices=list(CP_PVAL_MAP.keys()), required=True,
                   help="Which changepoint test to use: pettitt, stc, or var")

    p.add_argument("--suffix", default=None,
                   help="Optional suffix for inferred datasets (e.g. breakpoint_stc).")

    p.add_argument("--sm_path", default=None, help="Optional override path to soil moisture changepoint dataset")
    p.add_argument("--et_path", default=None, help="Optional override path to transpiration changepoint dataset")
    p.add_argument("--p_path",  default=None, help="Optional override path to precipitation changepoint dataset")

    p.add_argument("--outdir", default=None, help="Optional override output directory")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))

    def _sfx(s):
        if not s:
            return ""
        s = str(s).strip()
        return s if s.startswith("_") else f"_{s}"

    default_sm = outputs_root / "zarr" / f"out_sm{_sfx(args.suffix)}_chp.zarr"
    default_et = outputs_root / "zarr" / f"out_Et{_sfx(args.suffix)}_chp.zarr"
    default_p  = outputs_root / "zarr" / f"out_precip{_sfx(args.suffix)}_chp.zarr"
    default_outdir = outputs_root / "figures" / "abrupt_shift_cumulative"

    sm_path = Path(args.sm_path) if args.sm_path else default_sm
    et_path = Path(args.et_path) if args.et_path else default_et
    p_path  = Path(args.p_path)  if args.p_path  else default_p
    outdir  = Path(args.outdir)  if args.outdir  else default_outdir

    plot_accumulated_three(str(sm_path), str(et_path), str(p_path), args.test, outdir=str(outdir), cfg=cfg)


if __name__ == "__main__":
    plt.rc("figure", figsize=(13, 8))
    plt.rc("font", size=12)
    sn.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    
    main()