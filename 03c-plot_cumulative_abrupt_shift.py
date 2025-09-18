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

""" 
Plots and save cumulative land area with abrupt shifts for 
precipitation, transpiration, and soil moisture. 

E.g. 
    python3 03c-plot_cumulative_abrupt_shift.py --sm_path "/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr" --et_path "/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr" --p_path "/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr" --test 'stc' --outdir '/mnt/data/romi/figures/paper_1/supplementary_final/supp_abrupt_shifts'

"""

# -------------------- defaults --------------------
SM_PATH = "/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr"
ET_PATH = "/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr"
P_PATH  = "/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr"
LANDSEA_PATH = "/mnt/data/romi/data/landsea_mask.grib"

START_YEAR = 2000
WEEKS_PER_YEAR = 52.1775

# -------------------- cp/pval mapping --------------------
CP_PVAL_MAP = {
    "pettitt": {"cp": "pettitt_cp",     "pval": "pettitt_pval", "label": "Pettitt"},
    "stc":     {"cp": "strucchange_bp", "pval": "Fstat_pval",   "label": "StructChange F"},
    "var":     {"cp": "bp_var",         "pval": "pval_var",     "label": "Variance break"},
}

# -------------------- helpers --------------------
def open_source_ds(path: str) -> xr.Dataset:
    # handle zarr vs netcdf
    if os.path.isdir(path) and (os.path.exists(os.path.join(path, ".zgroup")) or
                                os.path.exists(os.path.join(path, ".zmetadata"))):
        return xr.open_zarr(path)
    return xr.open_dataset(path)

def lat_band_area_km2(lat_deg: np.ndarray, dlon: float = 0.25, dlat: float = 0.25) -> np.ndarray:
    """
    1D area per latitude (km^2) for a regular lon/lat grid, same style as your notebook.
    """
    R = 6371.0  # km
    lat_rad = np.radians(lat_deg)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)



def mask_precip_to_land(ds_p: xr.Dataset) -> xr.Dataset:
    """Return precip dataset masked to land (lsm>0.7) on ds_p's lon/lat grid."""
    
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

    # boolean land mask, rename dims to match precip
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
    
    # apply
    return ds_p.where(land)


def compute_accumulated_percent(ds: xr.Dataset, cp_name: str, pval_name: str,
                                start_year: int = START_YEAR,
                                weeks_per_year: float = WEEKS_PER_YEAR):
    """
    Returns:
        years (np.ndarray[int]): START_YEAR..max_year (inclusive)
        cumulative_percent (np.ndarray[float]): cumulative % land area with cp detected up to each year
    """
    if cp_name not in ds or pval_name not in ds:
        raise KeyError(f"Dataset missing variables: {cp_name} or {pval_name}")

    cp = ds[cp_name]          # expected 2D (lat, lon) or similar
    pval = ds[pval_name]

    # Only consider cells with a valid cp index and significant p-value
    sig = (pval < 0.05) & (cp > 0)
    if "lat" not in cp.dims or "lon" not in cp.dims:
        raise ValueError("Expected variables to have 'lat' and 'lon' dimensions.")

    # Area weights (broadcast 1D-by-lat to 2D lat x lon)
    lat = ds["lat"].values
    lon = ds["lon"].values
    area1d = lat_band_area_km2(lat, dlon=float(abs(np.median(np.diff(lon))) if lon.size > 1 else 0.25),
                                    dlat=float(abs(np.median(np.diff(lat))) if lat.size > 1 else 0.25))
    area2d = np.broadcast_to(area1d[:, None], (lat.size, lon.size))
    area_da = xr.DataArray(area2d, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    # Total valid land area baseline = where cp is not NaN
    valid = cp.notnull()
    total_area = float(area_da.where(valid).sum().values)
    if total_area <= 0:
        # No valid area -> return empty series
        return np.array([], dtype=int), np.array([], dtype=float)

    # Convert cp index (weeks since start) to calendar year (float), then to integer bins
    cp_year_float = start_year + (cp / weeks_per_year)
    cp_year_int = np.floor(cp_year_float).astype("float64")  # keep NaN for invalid/non-sig
    # keep only significant cells
    cp_year_int = cp_year_int.where(sig)

    # Determine year axis
    max_year = int(np.nanmax(cp_year_int.values)) if np.isfinite(cp_year_int.values).any() else start_year
    years = np.arange(start_year, max_year + 1, dtype=int)

    # Accumulate area by first-occurring cp year
    # For each year y, sum areas where cp_year_int == y
    cumulative_area = np.zeros_like(years, dtype=float)
    # Use raw numpy for speed
    cp_year_arr = cp_year_int.values
    area_arr = area_da.values

    for i, y in enumerate(years):
        mask_y = np.isfinite(cp_year_arr) & (cp_year_arr == float(y))
        yearly_area = area_arr[mask_y].sum()
        cumulative_area[i] = yearly_area if i == 0 else cumulative_area[i-1] + yearly_area

    cumulative_percent = 100.0 * cumulative_area / total_area
    return years, cumulative_percent

def extend_series_to(years_base: np.ndarray, years: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """
    Extend a cumulative series to a wider year axis by forward-filling the last value.
    """
    out = np.full_like(years_base, np.nan, dtype=float)
    # map existing values
    year_to_val = {int(y): float(v) for y, v in zip(years, vals)}
    last = 0.0
    for i, y in enumerate(years_base):
        if int(y) in year_to_val:
            last = year_to_val[int(y)]
        out[i] = last
    return out

# -------------------- plotting --------------------
def plot_accumulated_three(sm_ds_path: str, et_ds_path: str, p_ds_path: str,
                           test_name: str, outdir: str): 
    # Map test -> variable names
    if test_name not in CP_PVAL_MAP:
        raise ValueError(f"--test must be one of {list(CP_PVAL_MAP.keys())}")
    cp_name = CP_PVAL_MAP[test_name]["cp"]
    pval_name = CP_PVAL_MAP[test_name]["pval"]
    test_label = CP_PVAL_MAP[test_name]["label"]

    # Load datasets
    ds_sm = open_source_ds(sm_ds_path)
    ds_et = open_source_ds(et_ds_path)
    ds_p  = open_source_ds(p_ds_path)

    ds_p = mask_precip_to_land(ds_p)

    # Compute series
    y_sm, s_sm = compute_accumulated_percent(ds_sm, cp_name, pval_name)
    y_et, s_et = compute_accumulated_percent(ds_et, cp_name, pval_name)
    y_p,  s_p  = compute_accumulated_percent(ds_p,  cp_name, pval_name)

    # Build a common year axis
    max_year = max([y[-1] if y.size else START_YEAR for y in (y_sm, y_et, y_p)])
    years = np.arange(START_YEAR, max_year + 1, dtype=int)
    
    
    # Extend each series to the common axis by forward fill
    s_sm_e = extend_series_to(years, y_sm, s_sm) if y_sm.size else np.zeros_like(years, dtype=float)
    s_et_e = extend_series_to(years, y_et, s_et) if y_et.size else np.zeros_like(years, dtype=float)
    s_p_e  = extend_series_to(years, y_p,  s_p)  if y_p.size  else np.zeros_like(years, dtype=float)

    # COnver tot pandas to save
    df_sm = pd.Series(s_sm_e, index=years, name="SM")
    df_et = pd.Series(s_et_e, index=years, name="ET")
    df_p  = pd.Series(s_p_e,  index=years, name="P")

    df_sm.to_csv(os.path.join(outdir, f"values_sm_{test_name}.csv"))
    df_et.to_csv(os.path.join(outdir, f"values_et_{test_name}.csv"))
    df_p.to_csv(os.path.join(outdir, f"values_p_{test_name}.csv"))

    # Plot
    sn.set_style("whitegrid")
    plt.figure(figsize=(7, 3))
    plt.plot(years, s_sm_e, label="Soil moisture", linewidth=2, color = '#a58746')
    plt.plot(years, s_et_e, label="Transpiration", linewidth=2, color = "#845B7D")
    plt.plot(years, s_p_e,  label="Precipitation", linewidth=2, color = "#5FA7C0")

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
    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    os.makedirs(outdir, exist_ok=True)
    svg = os.path.join(outdir, f"accumulated_cp_area_{test_name}.svg")
    plt.savefig(svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Time series of accumulated land area with abrupt shifts for SM/ET/Precip.")
    ap.add_argument("--test", choices=list(CP_PVAL_MAP.keys()), required=True,
                    help="Which changepoint test to use: pettitt, stc, or var")
    
    ap.add_argument("--outdir", required=True, help="Output directory for the plot")
    ap.add_argument("--sm_path", default=SM_PATH, help="Path to soil moisture changepoint dataset")
    ap.add_argument("--et_path", default=ET_PATH, help="Path to evapotranspiration changepoint dataset")
    ap.add_argument("--p_path",  default=P_PATH,  help="Path to precipitation changepoint dataset")

    args = ap.parse_args()

    plot_accumulated_three(args.sm_path, args.et_path, args.p_path, args.test, outdir = args.outdir)

if __name__ == "__main__":
    
    plt.rc("figure", figsize=(13, 8))
    plt.rc("font", size=12)
    sn.set_style("white")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'


    main()