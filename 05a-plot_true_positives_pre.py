import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import regionmask
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


""" 

E.g. 
    python 05a-plot_true_positives_pre.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr' \
    --var 'sm' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' \
    --ews_pre_pos_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc_kt.zarr' \
    --ews_pre_neg_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc_neg_kt.zarr'

    python 05a-plot_true_positives_pre.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr' \
    --var 'Et' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' \
    --ews_pre_pos_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc_kt.zarr' \
    --ews_pre_neg_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc_neg_kt.zarr'

    python 05a-plot_true_positives_pre.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr' \
    --var 'precip' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' \
    --ews_pre_pos_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc_kt.zarr' \
    --ews_pre_neg_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc_neg_kt.zarr'
"""

# --- Paths & constants -------------------------------------------------------

TNC_SHP = "/mnt/data/romi/data/terr-ecoregions-TNC/tnc_terr_ecoregions.shp"

BIOMES = [
    'Tropical and Subtropical Moist Broadleaf Forests',
    'Tropical and Subtropical Dry Broadleaf Forests',
    'Tropical and Subtropical Coniferous Forests',
    'Temperate Broadleaf and Mixed Forests',
    'Temperate Conifer Forests',
    'Boreal Forests/Taiga',
    'Tropical and Subtropical Grasslands, Savannas and Shrublands',
    'Temperate Grasslands, Savannas and Shrublands',
    'Flooded Grasslands and Savannas',
    'Montane Grasslands and Shrublands',
    'Tundra',
    'Mediterranean Forests, Woodlands and Scrub',
    'Deserts and Xeric Shrublands',
    'Mangroves'
]

GROUP_MAP = {
    "Tropical and Subtropical Moist Broadleaf Forests":        "Tropical forests",
    "Tropical and Subtropical Dry Broadleaf Forests":          "Tropical forests",
    "Tropical and Subtropical Coniferous Forests":             "Tropical forests",
    "Temperate Broadleaf and Mixed Forests":                   "Temperate forests",
    "Temperate Conifer Forests":                               "Temperate forests",
    "Mediterranean Forests, Woodlands and Scrub":              "Mediterranean woodlands & scrub",
    "Boreal Forests/Taiga":                                    "Boreal forest / Taiga",
    "Tropical and Subtropical Grasslands, Savannas and Shrublands": "Tropical savannas & grasslands",
    "Temperate Grasslands, Savannas and Shrublands":                "Temperate grasslands & shrublands",
    "Montane Grasslands and Shrublands":                             "Montane grasslands & shrublands",
    "Deserts and Xeric Shrublands":                            "Deserts & xeric",
    "Tundra":                                                  "Tundra",
    "Cropland":                                                "Cropland",
}

groups_plot = [
    "Tropical forests",
    "Temperate forests",
    "Mediterranean woodlands & scrub",
    "Boreal forest / Taiga",
    "Tropical savannas & grasslands",
    "Temperate grasslands & shrublands",
    "Montane grasslands & shrublands",
    "Deserts & xeric",
    "Tundra",
    "Cropland",
]

ARIDITY_BINS   = [0, 0.05, 0.20, 0.50, 0.65, np.inf]
ARIDITY_LABELS = ["Hyper-arid", "Arid", "Semi-arid", "Dry subhumid", "Humid"]
ARIDITY_PLOT   = ARIDITY_LABELS

# --- Utilities ---------------------------------------------------------------

def detect_lon_lat_names(ds):
    lon_name = next((c for c in ["lon", "longitude", "x"] if c in ds.coords), None)
    lat_name = next((c for c in ["lat", "latitude", "y"] if c in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Could not detect lon/lat in coords: {list(ds.coords)}")
    return lon_name, lat_name

def wrap_to_180(ds, lon_name="lon"):
    return ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)

def apply_land_mask_if_precip(ds_breaks, ds_ews, var_name, lon_name, lat_name):
    if var_name != "precip":
        return ds_breaks, ds_ews
    ds_mask = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")
    if "longitude" in ds_mask.coords:
        ds_mask = ds_mask.assign_coords(longitude=(((ds_mask.longitude + 180) % 360) - 180)).sortby("longitude")
    ds_mask_interp = ds_mask.interp(
        longitude=ds_ews[lon_name],
        latitude=ds_ews[lat_name],
        method="nearest"
    )
    mask = ds_mask_interp["lsm"] > 0.7
    return ds_breaks.where(mask), ds_ews.where(mask)

def make_land_mask(ds_ews, lon_name, lat_name):
    land_poly = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_idx  = land_poly.mask(ds_ews[lon_name], ds_ews[lat_name])
    return xr.DataArray(np.isfinite(land_idx), coords=land_idx.coords, dims=land_idx.dims)

def get_biomes(biome):
    return gpd.read_file(TNC_SHP, where=f"WWF_MHTNAM='{biome}'")

def clip_biomes(ds, gdf):
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.set_spatial_dims("lon", "lat")
    return ds.rio.clip(gdf.geometry, gdf.crs, drop=True)

def get_area_grid(lat, dlon=0.25, dlat=0.25):
    R = 6371
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)

def area_da_for(clipped_ds):
    latitudes = clipped_ds['lat'].values
    area_per_lat = get_area_grid(latitudes)
    area_grid = np.broadcast_to(area_per_lat[:, np.newaxis], (latitudes.size, clipped_ds['lon'].size))
    return xr.DataArray(area_grid, coords={'lat': latitudes, 'lon': clipped_ds['lon'].values}, dims=('lat', 'lon'))

def _prepare_urban_crop_masks_like(ds_like):
    urban_mask = xr.open_dataset('/mnt/data/romi/data/urban_mask.zarr')
    crop_mask  = xr.open_dataset('/mnt/data/romi/data/crop_mask.zarr')
    urban_mask = urban_mask.rio.write_crs("EPSG:4326").interp_like(ds_like, method="nearest")
    crop_mask  = crop_mask.rio.write_crs("EPSG:4326").interp_like(ds_like, method="nearest")
    urban_mask = urban_mask['urban-coverfraction'].squeeze('time', drop=True)
    crop_mask  = crop_mask['crops-coverfraction'].squeeze('time', drop=True)
    return urban_mask, crop_mask

def _biome_group_iter(ds_base, urban_mask, crop_mask):
    for biome in BIOMES:
        if biome == "Mangroves":
            continue
        grp = GROUP_MAP.get(biome)
        if grp is None:
            continue
        gdf = get_biomes(biome)
        clipped = clip_biomes(ds_base, gdf)
        crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())
        valid_biome_mask = ((urban_mask <= 3) | urban_mask.isnull()) & (~crop_only)
        clipped = clipped.where(valid_biome_mask)
        yield grp, clipped
    grp = "Cropland"
    crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())
    yield grp, ds_base.where(crop_only)

def get_area_da(lat_vals, lon_vals):
    R = 6371.0
    dlat = float(np.abs(np.diff(lat_vals).mean())) if len(lat_vals) > 1 else 0.25
    dlon = float(np.abs(np.diff(lon_vals).mean())) if len(lon_vals) > 1 else 0.25
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    lat_rad = np.radians(lat_vals)
    row = (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
    grid = np.broadcast_to(row[:, None], (lat_vals.size, lon_vals.size))
    return xr.DataArray(grid, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))

def compute_aridity_classes_like(ds_target):
    precip = xr.open_dataset("/mnt/data/romi/data/ERA5_0.25_monthly/total_precipitation/total_precipitation_monthly.nc")\
              .sel(time=slice("2000-01-01", "2023-12-31"))\
              .rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    precip = wrap_to_180(precip, "lon")
    precip.rio.set_spatial_dims("lon", "lat", inplace=True)
    precip_tp = precip["tp"] * 1000.0
    pet = xr.open_dataset("/mnt/data/romi/data/et_pot/monthly_sum_epot_clean.zarr")\
             .sel(time=slice("2000-01-01", "2023-11-30"))
    precip["time"] = ("time", pd.date_range(str(precip.time.values[0])[:10], periods=precip.sizes["time"], freq="MS"))
    pet   ["time"] = ("time", pd.date_range(str(pet.time.values[0])[:10],     periods=pet.sizes["time"],     freq="MS"))
    lsm = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")\
            .rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = wrap_to_180(lsm, "lon")
    lsm.rio.set_spatial_dims("lon", "lat", inplace=True)
    lsm_on_precip = lsm["lsm"].interp(lat=precip.lat, lon=precip.lon)
    precip_tp = precip_tp.where(lsm_on_precip > 0.5)
    precip_aligned, pet_aligned = xr.align(precip_tp.where(precip_tp != 0, 1e-6),
                                           pet["pet"].where(pet["pet"] != 0, 1e-6),
                                           join="inner")
    aridity_mean  = (precip_aligned / pet_aligned).mean(dim="time").rename("aridity_index").where(lsm_on_precip > 0.5)
    lonn, latt = detect_lon_lat_names(ds_target)
    aridity_on_target = aridity_mean.interp(lat=ds_target[latt], lon=ds_target[lonn])
    class_index = xr.apply_ufunc(np.digitize, aridity_on_target, kwargs={"bins": ARIDITY_BINS},
                                 vectorize=True, dask="parallelized", output_dtypes=[int])
    def idx_to_label(idx):
        if 1 <= idx <= len(ARIDITY_LABELS): return ARIDITY_LABELS[idx-1]
        return np.nan
    label_array = xr.apply_ufunc(np.vectorize(idx_to_label), class_index,
                                 vectorize=True, dask="parallelized", output_dtypes=[object]).rename("aridity_class")
    if lonn != "lon" or latt != "lat":
        label_array = label_array.rename({lonn: "lon", latt: "lat"})
    return label_array

# --- Confusion helpers -------------------------------------------------------

def compute_confusion_arrays(ews_bool, bp_bool):
    e = ews_bool.astype(bool); b = bp_bool.astype(bool)
    TP = e & b; FP = e & ~b; FN = ~e & b; TN = ~e & ~b
    return TP, FP, FN, TN

def _counts_from_masks_area(e_mask, b_mask, area_da):
    TP, FP, FN, TN = compute_confusion_arrays(e_mask, b_mask)
    TP_a = float(area_da.where(TP).sum().values)
    FP_a = float(area_da.where(FP).sum().values)
    FN_a = float(area_da.where(FN).sum().values)
    TN_a = float(area_da.where(TN).sum().values)
    return TP_a, FP_a, FN_a, TN_a

# --- EWS masks from datasets (pre-τ) ----------------------------------------

def _ews_mask_from_ds(ds, label, var_name, alpha=0.05):
    """Build boolean EWS mask for a composite from a dataset with *_kt and *_pval fields."""
    def dir_mask(ds_loc, var, direction):
        tau = ds_loc[f'{var}_kt']; p = ds_loc[f'{var}_pval']
        return ((p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)).fillna(False)

    if label == 'FD ↓':
        tau = ds[f'{var_name}_fd_kt']; p = ds[f'{var_name}_fd_pval']
        return ((p < alpha) & (tau < 0)).fillna(False)

    if label == 'Skew ↑ & Kurt ↑':
        return (dir_mask(ds, f'{var_name}_skew','inc') &
                dir_mask(ds, f'{var_name}_kurt','inc')).fillna(False)

    if label == 'AC1 ↑ & SD ↑':
        return (dir_mask(ds, f'{var_name}_ac1','inc') &
                dir_mask(ds, f'{var_name}_std','inc')).fillna(False)

    if label == 'AC1 ↓ & SD ↓':
        return (dir_mask(ds, f'{var_name}_ac1','dec') &
                dir_mask(ds, f'{var_name}_std','dec')).fillna(False)

    if label == 'AC1 ↑↓ & SD ↑↓':
        return ((dir_mask(ds, f'{var_name}_ac1','inc') & dir_mask(ds, f'{var_name}_std','dec')) |
                (dir_mask(ds, f'{var_name}_ac1','dec') & dir_mask(ds, f'{var_name}_std','inc'))).fillna(False)

    raise ValueError(f"Unknown composite label: {label}")

# --- Fraction tables (area-weighted) ----------------------------------------

def _biome_confusion_fraction_table(ds_breaks, ds_full_kt, var_name,
                                    ds_pre_pos=None, ds_pre_neg=None):
    """Composite-only, method set inside heatmap later; returns long table for TP/FP/FN/TN by biome."""
    alpha = 0.05
    composites = ['AC1 ↑ & SD ↑', 'AC1 ↓ & SD ↓', 'AC1 ↑↓ & SD ↑↓', 'Skew ↑ & Kurt ↑', 'FD ↓']
    urban_mask, crop_mask = _prepare_urban_crop_masks_like(ds_full_kt)
    rows = []

    for grp, clipped_full in _biome_group_iter(ds_full_kt, urban_mask, crop_mask):
        if ('lat' not in clipped_full.coords) or ('lon' not in clipped_full.coords) or \
           clipped_full['lat'].size == 0 or clipped_full['lon'].size == 0:
            continue
        area_da = area_da_for(clipped_full)

        # pull break info (we only keep 'stc' downstream, but compute once here)
        bp_pv = ds_breaks['Fstat_pval'].interp_like(clipped_full, method="nearest")
        bp_cp = ds_breaks['strucchange_bp'].interp_like(clipped_full, method="nearest")
        b_mask_full = ((bp_pv < alpha) & (bp_cp > 0)).fillna(False)

        # if pre-τ provided, interpolate them to the same clipped grid
        clipped_pos = ds_pre_pos.interp_like(clipped_full, method="nearest") if ds_pre_pos is not None else None
        clipped_neg = ds_pre_neg.interp_like(clipped_full, method="nearest") if ds_pre_neg is not None else None

        for label in composites:
            if (clipped_pos is not None) and (clipped_neg is not None):
                e_pos = _ews_mask_from_ds(clipped_pos, label, var_name, alpha)
                e_neg = _ews_mask_from_ds(clipped_neg, label, var_name, alpha)
                avail_pos = e_pos.notnull()
                avail_neg = e_neg.notnull()
                eval_mask = (b_mask_full & avail_pos) | ((~b_mask_full) & avail_neg)
                e_mask = ((e_pos & b_mask_full) | (e_neg & (~b_mask_full))).where(eval_mask, False)
                b_mask = b_mask_full.where(eval_mask, False)
            else:
                # fallback to full τ (not recommended, but keeps function usable)
                def dir_mask(ds_loc, var, direction):
                    tau = ds_loc[f'{var}_kt']; p = ds_loc[f'{var}_pval']
                    return ((p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)).fillna(False)
                if label == 'FD ↓':
                    tau = clipped_full[f'{var_name}_fd_kt']; p = clipped_full[f'{var_name}_fd_pval']
                    e_mask = ((p < alpha) & (tau < 0)).fillna(False)
                elif label == 'Skew ↑ & Kurt ↑':
                    e_mask = (dir_mask(clipped_full,f'{var_name}_skew','inc') &
                              dir_mask(clipped_full,f'{var_name}_kurt','inc')).fillna(False)
                elif label == 'AC1 ↑ & SD ↑':
                    e_mask = (dir_mask(clipped_full,f'{var_name}_ac1','inc') &
                              dir_mask(clipped_full,f'{var_name}_std','inc')).fillna(False)
                elif label == 'AC1 ↓ & SD ↓':
                    e_mask = (dir_mask(clipped_full,f'{var_name}_ac1','dec') &
                              dir_mask(clipped_full,f'{var_name}_std','dec')).fillna(False)
                else:
                    e_mask = ((dir_mask(clipped_full,f'{var_name}_ac1','inc') &
                               dir_mask(clipped_full,f'{var_name}_std','dec')) |
                              (dir_mask(clipped_full,f'{var_name}_ac1','dec') &
                               dir_mask(clipped_full,f'{var_name}_std','inc'))).fillna(False)
                b_mask = b_mask_full

            TP_a, FP_a, FN_a, TN_a = _counts_from_masks_area(e_mask, b_mask, area_da)
            tot = TP_a + FP_a + FN_a + TN_a
            if tot <= 0:  # nothing to evaluate
                continue
            rows.extend([
                {"biome_group": grp, "method": "stc", "label": label, "cls": "TP", "frac": TP_a / tot},
                {"biome_group": grp, "method": "stc", "label": label, "cls": "FP", "frac": FP_a / tot},
                {"biome_group": grp, "method": "stc", "label": label, "cls": "FN", "frac": FN_a / tot},
                {"biome_group": grp, "method": "stc", "label": label, "cls": "TN", "frac": TN_a / tot},
            ])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["biome_group"] = pd.Categorical(df["biome_group"], categories=groups_plot, ordered=True)
    return df.sort_values(["biome_group", "method", "label", "cls"])

def _aridity_confusion_fraction_table(ds_breaks, ds_full_kt, var_name,
                                      ds_pre_pos=None, ds_pre_neg=None):
    """Composite-only, method 'stc'; returns long table for TP/FP/FN/TN by aridity class."""
    alpha = 0.05
    composites = ['AC1 ↑ & SD ↑', 'AC1 ↓ & SD ↓', 'AC1 ↑↓ & SD ↑↓', 'Skew ↑ & Kurt ↑', 'FD ↓']

    # valid terrestrial, non-urban mask on ds grid
    urban = xr.open_dataset("/mnt/data/romi/data/urban_mask.zarr").rio.write_crs("EPSG:4326").interp_like(ds_full_kt, method="nearest")
    urban = urban["urban-coverfraction"].squeeze("time", drop=True)
    lsm   = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib").rename({"latitude":"lat","longitude":"lon"}).rio.write_crs("EPSG:4326")
    lsm   = wrap_to_180(lsm, "lon"); lsm.rio.set_spatial_dims("lon","lat", inplace=True)
    lsm_on_ds = lsm["lsm"].interp(lat=ds_full_kt["lat"], lon=ds_full_kt["lon"])
    valid_mask = ((lsm_on_ds > 0.5) | lsm_on_ds.isnull()) & ((urban <= 3) | urban.isnull())

    aridity_class = compute_aridity_classes_like(ds_full_kt).where(valid_mask)
    area_da = get_area_da(ds_full_kt["lat"].values, ds_full_kt["lon"].values).where(valid_mask)

    bp_pv = ds_breaks['Fstat_pval']; bp_cp = ds_breaks['strucchange_bp']
    b_mask_full = ((bp_pv < alpha) & (bp_cp > 0)).fillna(False)

    rows = []
    for label in composites:
        if (ds_pre_pos is not None) and (ds_pre_neg is not None):
            e_pos = _ews_mask_from_ds(ds_pre_pos, label, var_name, alpha)
            e_neg = _ews_mask_from_ds(ds_pre_neg, label, var_name, alpha)
            avail_pos = e_pos.notnull()
            avail_neg = e_neg.notnull()
            eval_mask_full = (b_mask_full & avail_pos) | ((~b_mask_full) & avail_neg)
            e_mask_full = ((e_pos & b_mask_full) | (e_neg & (~b_mask_full))).where(eval_mask_full, False)
            b_mask_full_eval = b_mask_full.where(eval_mask_full, False)
        else:
            # fallback to full τ
            def dir_mask(ds_loc, var, direction):
                tau = ds_loc[f'{var}_kt']; p = ds_loc[f'{var}_pval']
                return ((p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)).fillna(False)
            if label == 'FD ↓':
                tau = ds_full_kt[f'{var_name}_fd_kt']; p = ds_full_kt[f'{var_name}_fd_pval']
                e_mask_full = ((p < alpha) & (tau < 0)).fillna(False)
            elif label == 'Skew ↑ & Kurt ↑':
                e_mask_full = (dir_mask(ds_full_kt,f'{var_name}_skew','inc') &
                               dir_mask(ds_full_kt,f'{var_name}_kurt','inc')).fillna(False)
            elif label == 'AC1 ↑ & SD ↑':
                e_mask_full = (dir_mask(ds_full_kt,f'{var_name}_ac1','inc') &
                               dir_mask(ds_full_kt,f'{var_name}_std','inc')).fillna(False)
            elif label == 'AC1 ↓ & SD ↓':
                e_mask_full = (dir_mask(ds_full_kt,f'{var_name}_ac1','dec') &
                               dir_mask(ds_full_kt,f'{var_name}_std','dec')).fillna(False)
            else:
                e_mask_full = ((dir_mask(ds_full_kt,f'{var_name}_ac1','inc') &
                                dir_mask(ds_full_kt,f'{var_name}_std','dec')) |
                               (dir_mask(ds_full_kt,f'{var_name}_ac1','dec') &
                                dir_mask(ds_full_kt,f'{var_name}_std','inc'))).fillna(False)
            b_mask_full_eval = b_mask_full

        for cls_label in ARIDITY_PLOT:
            cls_mask = (aridity_class == cls_label)
            if not bool(cls_mask.any()):
                continue
            area_cls = area_da.where(cls_mask)
            if float(area_cls.sum().values) == 0.0:
                continue

            TP, FP, FN, TN = compute_confusion_arrays(e_mask_full & cls_mask, b_mask_full_eval & cls_mask)
            TP_a = float(area_cls.where(TP).sum().values)
            FP_a = float(area_cls.where(FP).sum().values)
            FN_a = float(area_cls.where(FN).sum().values)
            TN_a = float(area_cls.where(TN).sum().values)
            tot  = TP_a + FP_a + FN_a + TN_a
            if tot <= 0:
                continue
            rows.extend([
                {"aridity_class": cls_label, "method": "stc", "label": label, "cls": "TP", "frac": TP_a / tot},
                {"aridity_class": cls_label, "method": "stc", "label": label, "cls": "FP", "frac": FP_a / tot},
                {"aridity_class": cls_label, "method": "stc", "label": label, "cls": "FN", "frac": FN_a / tot},
                {"aridity_class": cls_label, "method": "stc", "label": label, "cls": "TN", "frac": TN_a / tot},
            ])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["aridity_class"] = pd.Categorical(df["aridity_class"], categories=ARIDITY_PLOT, ordered=True)
    return df.sort_values(["method", "label", "aridity_class", "cls"])

# --- Balanced Accuracy heatmap (diverging palette) --------------------------

def _heatmap_balacc(df_long, index_col, index_order, labels_order, title, out_fp,
                    center_on="mean"):  # center_on: "mean" or 0.5
    """
    df_long columns: [<index_col>, 'label', 'cls', 'frac'] for method='stc' and composites.
    Builds (index_order x labels_order) matrix of Balanced Accuracy with cells set to NaN when
    no positives (TP+FN=0) or no negatives (TN+FP=0) in that cell.
    """
    def balanced_accuracy(tp, fp, fn, tn):
        pos = tp + fn; neg = tn + fp
        if pos == 0 or neg == 0:
            return np.nan
        return 0.5 * (tp/pos + tn/neg)

    recs, supports = [], []
    for idx in index_order:
        df_i = df_long[df_long[index_col] == idx]
        for lab in labels_order:
            df_il = df_i[df_i['label'] == lab]
            if df_il.empty:
                recs.append((idx, lab, np.nan)); supports.append((idx, lab, 0.0)); continue
            tp = float(df_il[df_il['cls'] == 'TP']['frac'].sum())
            fp = float(df_il[df_il['cls'] == 'FP']['frac'].sum())
            fn = float(df_il[df_il['cls'] == 'FN']['frac'].sum())
            tn = float(df_il[df_il['cls'] == 'TN']['frac'].sum())
            pos, neg = tp + fn, tn + fp
            if pos == 0 or neg == 0:
                recs.append((idx, lab, np.nan)); supports.append((idx, lab, 0.0)); continue
            ba = balanced_accuracy(tp, fp, fn, tn)
            recs.append((idx, lab, ba)); supports.append((idx, lab, pos + neg))

    mat = (pd.DataFrame(recs, columns=[index_col, 'label', 'BA'])
              .pivot(index=index_col, columns='label', values='BA')
              .reindex(index_order)[labels_order])

    # Colormap: red (#b97474ff) → white → blue (#85b6c5ff)
    cmap = LinearSegmentedColormap.from_list("ba_div", ["#b97474ff", "white", "#85b6c5ff"])

    # Center on mean of defined cells, or on 0.5 (random)
    if center_on == "mean":
        vcenter = float(np.nanmean(mat.values))
    else:
        vcenter = 0.5

    vmin = float(np.nanpercentile(mat.values, 1)) if np.isfinite(mat.values).any() else 0.4
    vmax = float(np.nanpercentile(mat.values, 99)) if np.isfinite(mat.values).any() else 0.6
    # Ensure sensible bounds around center
    vmin = min(vmin, vcenter - 0.1)
    vmax = max(vmax, vcenter + 0.1)

    mask = mat.isna()
    annot_str = mat.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(1.6*len(labels_order), 0.5*len(index_order) + 2))
    sn.heatmap(mat, mask=mask, cmap=cmap, norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
               annot=annot_str, fmt="", annot_kws={"fontsize": 9},
               cbar_kws={'label': 'Balanced accuracy'}, linewidths=0.3, linecolor='white', ax=ax)
    ax.set_xlabel('Composite EWS')
    ax.set_ylabel(index_col.replace('_', ' ').title())
    ax.set_title(title, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    fig.savefig(out_fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)

# --- Wrappers to build and plot the two heatmaps ----------------------------

def plot_biome_balacc_heatmap_stc(ds_breaks, ds_full_kt, var_name, out_dir, ds_pre_pos=None, ds_pre_neg=None):
    df_bio = _biome_confusion_fraction_table(ds_breaks, ds_full_kt, var_name,
                                             ds_pre_pos=ds_pre_pos, ds_pre_neg=ds_pre_neg)
    if df_bio.empty:
        return
    df_bio = df_bio[df_bio['method'] == 'stc'].copy()
    labels_order = ['AC1 ↑ & SD ↑', 'AC1 ↓ & SD ↓', 'AC1 ↑↓ & SD ↑↓', 'Skew ↑ & Kurt ↑', 'FD ↓']
    title = f"{var_name}: Balanced accuracy by biome (composites, stc)"
    out_fp = os.path.join(out_dir, f"{var_name}_stc_biome_heatmap_composites_PRE.svg")
    _heatmap_balacc(df_bio, index_col='biome_group', index_order=groups_plot,
                    labels_order=labels_order, title=title, out_fp=out_fp, center_on="mean")

def plot_aridity_balacc_heatmap_stc(ds_breaks, ds_full_kt, var_name, out_dir, ds_pre_pos=None, ds_pre_neg=None):
    df_arid = _aridity_confusion_fraction_table(ds_breaks, ds_full_kt, var_name,
                                                ds_pre_pos=ds_pre_pos, ds_pre_neg=ds_pre_neg)
    if df_arid.empty:
        return
    df_arid = df_arid[df_arid['method'] == 'stc'].copy()
    labels_order = ['AC1 ↑ & SD ↑', 'AC1 ↓ & SD ↓', 'AC1 ↑↓ & SD ↑↓', 'Skew ↑ & Kurt ↑', 'FD ↓']
    title = f"{var_name}: Balanced accuracy by aridity class (composites, stc)"
    out_fp = os.path.join(out_dir, f"{var_name}_stc_aridity_heatmap_composites_PRE.svg")
    _heatmap_balacc(df_arid, index_col='aridity_class', index_order=ARIDITY_PLOT,
                    labels_order=labels_order, title=title, out_fp=out_fp, center_on="mean")

# --- Main -------------------------------------------------------------------

def main():
    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    parser = argparse.ArgumentParser(description='Balanced Accuracy heatmaps (composite EWS vs stc) by biome & aridity.')
    parser.add_argument('--ews_kt_path', type=str, required=True, help='Path to full-series EWS KT dataset (grid/template).')
    parser.add_argument('--ds_cp_path', type=str, required=True, help='Path to changepoint dataset (must include strucchange_bp, Fstat_pval).')
    parser.add_argument('--var', type=str, required=True, help='Variable name (e.g., Et, precip, sm).')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for figures.')
    parser.add_argument('--ews_pre_pos_path', type=str, default=None, help='(Optional) pre-break τ dataset for positives (tau_pre).')
    parser.add_argument('--ews_pre_neg_path', type=str, default=None, help='(Optional) pre-break τ dataset for negatives (tau_pre_neg).')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds_full = xr.open_dataset(args.ews_kt_path)
    ds_cp   = xr.open_dataset(args.ds_cp_path)

    # optional pre-τ datasets
    ds_pre_pos = xr.open_dataset(args.ews_pre_pos_path) if args.ews_pre_pos_path else None
    ds_pre_neg = xr.open_dataset(args.ews_pre_neg_path) if args.ews_pre_neg_path else None

    # If precip, apply land mask consistently
    lon_name, lat_name = detect_lon_lat_names(ds_full)
    ds_cp, ds_full = apply_land_mask_if_precip(ds_cp, ds_full, args.var, lon_name, lat_name)
    if ds_pre_pos is not None:
        ds_pre_pos = ds_pre_pos.interp_like(ds_full, method="nearest").where(ds_full[list(ds_full.data_vars)[0]].notnull())
    if ds_pre_neg is not None:
        ds_pre_neg = ds_pre_neg.interp_like(ds_full, method="nearest").where(ds_full[list(ds_full.data_vars)[0]].notnull())

    print('--- Balanced Accuracy heatmap by biome (composites, stc) ---')
    plot_biome_balacc_heatmap_stc(ds_breaks=ds_cp, ds_full_kt=ds_full, var_name=args.var,
                                  out_dir=args.out_dir, ds_pre_pos=ds_pre_pos, ds_pre_neg=ds_pre_neg)

    print('--- Balanced Accuracy heatmap by aridity class (composites, stc) ---')
    plot_aridity_balacc_heatmap_stc(ds_breaks=ds_cp, ds_full_kt=ds_full, var_name=args.var,
                                    out_dir=args.out_dir, ds_pre_pos=ds_pre_pos, ds_pre_neg=ds_pre_neg)

    print('--- Done ---')

if __name__ == '__main__':
    main()
