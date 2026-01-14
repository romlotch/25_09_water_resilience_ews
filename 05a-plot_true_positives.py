import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import regionmask
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.config import load_config, cfg_path, cfg_get


""" 

EDIT: other scores, not just F1

Plots and save F1 scores of both indicators and EWS for each of the changepoint methods. 

Plots and save of the spatial classification of performance of both indicators and EWS with respect 
to the changepoints analysis, for each of the changepoint methods. 

Inputs:
    --ews_kt_path: Path to the kendall tau's of the ews output file. 
    --ds_cp_path: Path to the changepoint output file.
    --var: Variable name in the kendall tau dataset (e.g. Et, precip, sm)
    --out_dir: Directory to save output figures to 

E.g. 
    python 05a-plot_true_positives.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm_chp.zarr' \
    --var 'sm' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' 

    python 05a-plot_true_positives.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et_chp.zarr' \
    --var 'Et' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' 

    python 05a-plot_true_positives.py \
    --ews_kt_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr' \
    --ds_cp_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip_chp.zarr' \
    --var 'precip' \
    --out_dir '/mnt/data/romi/figures/paper_1/results_final/figure_4' 

"""

# --- Hardcoded biome bits --- 

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

BIOMES_PLOT = [
    'Tropical and Subtropical Moist Broadleaf Forests',
    'Tropical and Subtropical Dry Broadleaf Forests',
    'Tropical and Subtropical Coniferous Forests',
    'Tropical and Subtropical Grasslands, Savannas and Shrublands',
    'Deserts and Xeric Shrublands',
    'Mediterranean Forests, Woodlands and Scrub',
    'Temperate Grasslands, Savannas and Shrublands',
    'Montane Grasslands and Shrublands',
    'Temperate Broadleaf and Mixed Forests',
    'Temperate Conifer Forests',
    'Boreal Forests/Taiga',
    'Tundra',
    'Flooded Grasslands and Savannas',
    'Mangroves',
    'Cropland'
]

INDICATORS_META = [
    ("ac1",  "AC1"),
    ("std",  "SD"),
    ("fd",   "FD"),
    ("skew", "Skew"),
    ("kurt", "Kurt"),
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
    # "Flooded Grasslands and Savannas":                       "Flooded grasslands & savannas",

    "Cropland":                                                "Cropland",
    # "Mangroves":                                             # excluded
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
    # "Flooded grasslands & savannas",
    "Tundra",
    "Cropland",
]

# --- Aridity bins (5-bins) ---
ARIDITY_BINS = [0, 0.05, 0.20, 0.50, 0.65, np.inf]
ARIDITY_LABELS = [
    "Hyper-arid",
    "Arid",
    "Semi-arid",
    "Dry subhumid",
    "Humid",
]
ARIDITY_PLOT = ARIDITY_LABELS


# --- Helpers ---

def detect_lon_lat_names(ds):
    """Return (lon_name, lat_name) from common candidates."""
    lon_name = next((c for c in ["lon", "longitude", "x"] if c in ds.coords), None)
    lat_name = next((c for c in ["lat", "latitude", "y"] if c in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Could not detect lon/lat in coords: {list(ds.coords)}")
    return lon_name, lat_name

def wrap_to_180(ds, lon_name="lon"):
    ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
    return ds

def apply_land_mask_if_precip(ds_breaks, ds_ews, var_name, lon_name, lat_name, cfg):
    """If var is 'precip', apply land-sea mask (lsm > 0.7) to BOTH! datasets so theyre consistent."""
    if var_name != "precip":
        return ds_breaks, ds_ews

    mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
    ds_mask = xr.open_dataset(mask_path, engine ="cfgrib")
    
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
    """Regionmask land mask on the EWS grid."""
    land_poly = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_idx  = land_poly.mask(ds_ews[lon_name], ds_ews[lat_name])
    return xr.DataArray(np.isfinite(land_idx), coords=land_idx.coords, dims=land_idx.dims)

def clip_antarctica(da):
    """Clip below -60 if a latitude coordinate exists."""
    lat_key = next((c for c in ["lat", "latitude", "y"] if c in da.coords), None)
    return da if lat_key is None else da.where(da[lat_key] >= -60)

def safe_label(s: str) -> str:
    return s.replace(" ", "_").replace("↑", "up").replace("↓", "down").replace("/", "-")


def compute_confusion_arrays(ews_bool, bp_bool):
    """Return TP, FP, FN, TN boolean arrays on the same grid."""
    e = ews_bool.astype(bool)
    b = bp_bool.astype(bool)
    TP = e & b
    FP = e & ~b
    FN = ~e & b
    TN = ~e & ~b
    return TP, FP, FN, TN

def class_map_from_confusion(TP, FP, FN, TN):
    """Return class map with 1=TP, 2=FP, 3=FN, 4=TN, NaN elsewhere."""
    return xr.where(TP, 1,
           xr.where(FP, 2,
           xr.where(FN, 3,
           xr.where(TN, 4, np.nan))))

def class_colors_norm():
    colors = {1: '#7d475dff', 2: '#ffeaefff', 3: '#e9e2bcff', 4: '#2c787eff'}
    cmap = ListedColormap([colors[i] for i in range(1, 5)])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ncolors=4)
    return colors, cmap, norm


# --- Maps and global bar plots ---

CLASS_LABELS = {1: 'TP', 2: 'FP', 3: 'FN', 4: 'TN'}
CLASS_COLORS = {1: '#7d475dff', 2: '#ffeaefff', 3: '#e9e2bcff', 4: '#2c787eff'}

def plot_class_map(da, title):
    _, cmap, norm = class_colors_norm()
    da_plot = clip_antarctica(da)
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson()})
    da_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    ax.coastlines(linewidth=0.5)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.patch.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show()
    return fig

def plot_class_bar(cmap_da, title):
    colors, _, _ = class_colors_norm()
    labels = {1: 'TP', 2: 'FP', 3: 'FN', 4: 'TN'}
    vals = cmap_da.values.ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        vals = np.array([])
    unique, counts = np.unique(vals, return_counts=True)
    total = counts.sum() if counts.size else 1
    perc = {int(k): (v / total) for k, v in zip(unique, counts)}

    classes   = [1, 2, 3, 4]
    perc_list = [perc.get(c, 0) for c in classes]
    ticklabs  = [labels[c] for c in classes]
    cols      = [colors[c] for c in classes]
    y_pos     = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(y_pos, perc_list, color=cols, height=0.9)
    for bar, frac in zip(bars, perc_list):
        x = bar.get_width()
        ax.text(x + 0.01, bar.get_y() + bar.get_height()/2,
                f"{frac:.1%}", va='center', ha='left', fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ticklabs)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of pixels')
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig


# --- Performance metrics ---

def metric_scores(TP, FP, FN, TN):
    TPc, FPc, FNc, TNc = TP, FP, FN, TN
    denom_f1 = (2*TPc + FPc + FNc)
    f1 = (2*TPc) / denom_f1 if denom_f1 > 0 else np.nan
    prec = TPc / (TPc + FPc) if (TPc + FPc) > 0 else np.nan
    acc = (TPc + TNc) / (TPc + FPc + FNc + TNc) if (TPc + FPc + FNc + TNc) > 0 else np.nan
    return {"F1": f1, "Precision": prec, "Accuracy": acc}

def compute_global_counts(e_mask, b_mask):
    TP, FP, FN, TN = compute_confusion_arrays(e_mask, b_mask)
    return int(TP.sum()), int(FP.sum()), int(FN.sum()), int(TN.sum())

def _global_masks_and_methods(ds_breaks, ds_ews, var_name):
    alpha = 0.05
    indicators = [f'{var_name}_ac1', f'{var_name}_std', f'{var_name}_fd',
                  f'{var_name}_skew', f'{var_name}_kurt']
    methods = {
        'pettitt':  {'cp': ds_breaks['pettitt_cp'],     'pval': ds_breaks['pettitt_pval']},
        'stc':      {'cp': ds_breaks['strucchange_bp'], 'pval': ds_breaks['Fstat_pval']},
        'variance': {'cp': ds_breaks['bp_var'],         'pval': ds_breaks['pval_var']},
    }
    return alpha, indicators, methods

def _ews_mask_rule(ds, var_name, v, alpha, land_mask):
    tau = ds[f'{v}_kt']; p = ds[f'{v}_pval']
    base = (p < alpha) & tau.notnull()
    if v.endswith('_skew') or v.endswith('_kurt'):
        base &= (tau > 0)
    if v.endswith('_fd'):
        base &= (tau < 0)
    return base.where(land_mask, False).fillna(False)

def _break_mask(methods, method, land_mask, alpha):
    cp = methods[method]['cp']; p = methods[method]['pval']
    return ((p < alpha) & (cp > 0)).where(land_mask, False).fillna(False)

def _plot_metric_bar(df_results, x, ylabel, out_fp, method_order, colours):
    fig, ax = plt.subplots(figsize=(6, 3))
    sn.barplot(data=df_results, x=x, y='value',
               hue='method', hue_order=method_order, palette=colours, ax=ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.8)
    ax.grid(False, axis='x')
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.3),
              ncol=len(method_order), frameon=False)
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    fig.savefig(out_fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def _collect_metric_table_for_indicators(ds_breaks, ds_ews, var_name, metric="F1"):
    alpha, indicators, methods = _global_masks_and_methods(ds_breaks, ds_ews, var_name)
    lon_name, lat_name = detect_lon_lat_names(ds_ews)
    land_mask = make_land_mask(ds_ews, lon_name, lat_name)
    rows = []
    for v in indicators:
        e_mask = _ews_mask_rule(ds_ews, var_name, v, alpha, land_mask)
        for method in methods:
            b_mask = _break_mask(methods, method, land_mask, alpha)
            TP, FP, FN, TN = compute_global_counts(e_mask, b_mask)
            score = metric_scores(TP, FP, FN, TN)[metric]
            rows.append({'indicator': v, 'method': method, 'value': score})
    return pd.DataFrame(rows)

def _collect_metric_table_for_composites(ds_breaks, ds_ews, var_name, metric="F1"):
    alpha, _, methods = _global_masks_and_methods(ds_breaks, ds_ews, var_name)
    lon_name, lat_name = detect_lon_lat_names(ds_ews)
    land_mask = make_land_mask(ds_ews, lon_name, lat_name)
    def ews_dir_mask(ds, var, direction):
        tau = ds[f'{var}_kt']; p = ds[f'{var}_pval']
        m = (p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)
        return m.where(land_mask, False).fillna(False)
    composites = {
        'AC1 ↑ & SD ↑':  lambda ds: ews_dir_mask(ds, f'{var_name}_ac1', 'inc') & ews_dir_mask(ds, f'{var_name}_std', 'inc'),
        'AC1 ↓ & SD ↓':  lambda ds: ews_dir_mask(ds, f'{var_name}_ac1', 'dec') & ews_dir_mask(ds, f'{var_name}_std', 'dec'),
        'AC1 ↑↓ & SD ↑↓':lambda ds: (
            (ews_dir_mask(ds, f'{var_name}_ac1', 'inc') & ews_dir_mask(ds, f'{var_name}_std', 'dec')) |
            (ews_dir_mask(ds, f'{var_name}_ac1', 'dec') & ews_dir_mask(ds, f'{var_name}_std', 'inc'))),
        'Skew ↑ & Kurt ↑': lambda ds: ews_dir_mask(ds, f'{var_name}_skew', 'inc') & ews_dir_mask(ds, f'{var_name}_kurt', 'inc'),
        'FD ↓':            lambda ds: ews_dir_mask(ds, f'{var_name}_fd', 'dec'),
    }
    rows = []
    for label, mask_fn in composites.items():
        e_mask = mask_fn(ds_ews)
        for method in methods:
            b_mask = _break_mask(methods, method, land_mask, alpha)
            TP, FP, FN, TN = compute_global_counts(e_mask, b_mask)
            score = metric_scores(TP, FP, FN, TN)[metric]
            rows.append({'composite': label, 'method': method, 'value': score})
    return pd.DataFrame(rows)

def plot_metric_indicators(ds_breaks, ds_ews, var_name, out_dir, metric="F1"):
    df = _collect_metric_table_for_indicators(ds_breaks, ds_ews, var_name, metric=metric)
    method_order = ['pettitt', 'stc', 'variance']
    colours      = ['#0072B2', '#E59D00', '#CC79A8']
    out_fp = f'{out_dir}/{var_name}_{metric.lower()}_score_indicators.svg'
    _plot_metric_bar(df, x='indicator', ylabel=f'{metric} Score', out_fp=out_fp,
                     method_order=method_order, colours=colours)

def plot_metric_ews(ds_breaks, ds_ews, var_name, out_dir, metric="F1"):
    df = _collect_metric_table_for_composites(ds_breaks, ds_ews, var_name, metric=metric)
    method_order = ['pettitt', 'stc', 'variance']
    colours      = ['#0072B2', '#E59D00', '#CC79A8']
    out_fp = f'{out_dir}/{var_name}_{metric.lower()}_score_ews.svg'
    _plot_metric_bar(df, x='composite', ylabel=f'{metric} Score', out_fp=out_fp,
                     method_order=method_order, colours=colours)


# --- Biome level scores ---

def get_biomes(biome):
    data = gpd.read_file(TNC_SHP, where=f"WWF_MHTNAM='{biome}'")
    return data


def clip_biomes(ds, gdf):
    ds = ds.rio.write_crs('EPSG:4326')
    ds = ds.rio.set_spatial_dims("lon", "lat")  # in-place behavior changed in rioxarray 0.15
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped


def get_area_grid(lat, dlon=0.25, dlat=0.25):
    R = 6371  # km
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)


def area_da_for(clipped_ds):
    """Broadcast 1D lat area to 2D (lat, lon)."""
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
   
    # Natural biomes
    for biome in BIOMES:
        if biome == "Mangroves":
            continue
        grp = GROUP_MAP.get(biome)
        if grp is None:
            continue
        gdf = get_biomes(biome)
        clipped = clip_biomes(ds_base, gdf)

        # non‑urban, non‑crop
        crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())
        valid_biome_mask = ((urban_mask <= 3) | urban_mask.isnull()) & (~crop_only)
        clipped = clipped.where(valid_biome_mask)

        yield grp, clipped

    # Cropland pseudo‑biome
    grp = "Cropland"
    crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())
    crop_clipped = ds_base.where(crop_only)
    yield grp, crop_clipped


def _counts_from_masks_area(e_mask, b_mask, area_da):
    """Area‑weighted counts for TP/FP/FN/TN in km2; returns floats."""
    TP, FP, FN, TN = compute_confusion_arrays(e_mask, b_mask)
    TP_a = float(area_da.where(TP).sum().values)
    FP_a = float(area_da.where(FP).sum().values)
    FN_a = float(area_da.where(FN).sum().values)
    TN_a = float(area_da.where(TN).sum().values)
    return TP_a, FP_a, FN_a, TN_a

# --- Biome confusion ---

def _biome_confusion_fraction_table(ds_breaks, ds_ews, var_name, which="indicators"):
    """
    Long-format DF with area-weighted fractions of TP/FP/FN/TN per biome group,
    for each indicator (or composite) and CP method.
    Columns: biome_group, method, label, cls, frac
    """
    alpha, indicators, _methods = _global_masks_and_methods(ds_breaks, ds_ews, var_name)

    def ews_dir_mask_local(ds_loc, var, direction):
        tau = ds_loc[f'{var}_kt']; p = ds_loc[f'{var}_pval']
        m = (p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)
        return m.fillna(False)

    composites = {
        'AC1 up & SD up':   lambda ds_loc: ews_dir_mask_local(ds_loc, f'{var_name}_ac1', 'inc') &
                                         ews_dir_mask_local(ds_loc, f'{var_name}_std', 'inc'),
        'AC1 down & SD down':   lambda ds_loc: ews_dir_mask_local(ds_loc, f'{var_name}_ac1', 'dec') &
                                         ews_dir_mask_local(ds_loc, f'{var_name}_std', 'dec'),
        'AC1 updown & SD updown': lambda ds_loc: (
            ews_dir_mask_local(ds_loc, f'{var_name}_ac1', 'inc') & ews_dir_mask_local(ds_loc, f'{var_name}_std', 'dec')
        ) | (
            ews_dir_mask_local(ds_loc, f'{var_name}_ac1', 'dec') & ews_dir_mask_local(ds_loc, f'{var_name}_std', 'inc')
        ),
        'Skew up & Kurt up': lambda ds_loc: ews_dir_mask_local(ds_loc, f'{var_name}_skew', 'inc') &
                                          ews_dir_mask_local(ds_loc, f'{var_name}_kurt', 'inc'),
        'FD down':            lambda ds_loc: ews_dir_mask_local(ds_loc, f'{var_name}_fd', 'dec'),
    }

    urban_mask, crop_mask = _prepare_urban_crop_masks_like(ds_ews)

    rows = []
    for grp, clipped in _biome_group_iter(ds_ews, urban_mask, crop_mask):
        if ('lat' not in clipped.coords) or ('lon' not in clipped.coords) or clipped['lat'].size == 0 or clipped['lon'].size == 0:
            continue
        area_da = area_da_for(clipped)

        # labels to iterate
        if which == "indicators":
            label_iter = [f'{var_name}_{k}' for k, _ in INDICATORS_META]
        else:
            label_iter = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']

        for label in label_iter:
            # Indicator or EWS mask
            if which == "indicators":
                tau = clipped[f'{label}_kt']; p = clipped[f'{label}_pval']
                base = (p < 0.05) & tau.notnull()
                if label.endswith("_skew") or label.endswith("_kurt"):
                    base &= (tau > 0)
                if label.endswith("_fd"):
                    base &= (tau < 0)
                e_mask = base.fillna(False)
            else:
                e_mask = composites[label](clipped).fillna(False)

            for method, mpair in {
                'pettitt':  {'cp': 'pettitt_cp', 'pval': 'pettitt_pval'},
                'stc':      {'cp': 'strucchange_bp', 'pval': 'Fstat_pval'},
                'variance': {'cp': 'bp_var', 'pval': 'pval_var'},
            }.items():
                cp = ds_breaks[mpair['cp']].interp_like(clipped, method="nearest")
                pv = ds_breaks[mpair['pval']].interp_like(clipped, method="nearest")
                b_mask = ((pv < 0.05) & (cp > 0)).fillna(False)

                TP_a, FP_a, FN_a, TN_a = _counts_from_masks_area(e_mask, b_mask, area_da)
                tot_a = TP_a + FP_a + FN_a + TN_a
                if tot_a <= 0:
                    continue
                rows.extend([
                    {"biome_group": grp, "method": method, "label": label, "cls": "TP", "frac": TP_a / tot_a},
                    {"biome_group": grp, "method": method, "label": label, "cls": "FP", "frac": FP_a / tot_a},
                    {"biome_group": grp, "method": method, "label": label, "cls": "FN", "frac": FN_a / tot_a},
                    {"biome_group": grp, "method": method, "label": label, "cls": "TN", "frac": TN_a / tot_a},
                ])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["biome_group"] = pd.Categorical(df["biome_group"], categories=groups_plot, ordered=True)
    return df.sort_values(["biome_group", "method", "label"])

def _indicator_pretty_name(full_var: str) -> str:
    suffix = full_var.split("_")[-1]
    lookup = dict(INDICATORS_META)
    return lookup.get(suffix, suffix.upper())

def plot_biome_confusion_rows(df_long, var_name, out_dir, which="indicators"):
    
    colors = {'TP': '#7d475dff', 'FP': '#ffeaefff', 'FN': '#e9e2bcff', 'TN': '#2c787eff'}
    classes = ['TP', 'FP', 'FN', 'TN']

    methods = ['pettitt', 'stc', 'variance']
    if which == "indicators":
        labels_order = [f"{var_name}_{k}" for k, _ in INDICATORS_META]
        titles       = [_indicator_pretty_name(lab) for lab in labels_order]
        fig_tag      = "indicators"
    else:
        labels_order = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']
        titles       = labels_order
        fig_tag      = "ews"

    for method in methods:
        df_m = df_long[df_long['method'] == method]
        if df_m.empty:
            continue

        sn.set_style("whitegrid")
        fig, axes = plt.subplots(1, 5, figsize=(3.9*5, 3.6), sharey=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, label, title in zip(axes, labels_order, titles):
            df_ml = df_m[df_m['label'] == label]
            if df_ml.empty:
                # draw empty axes
                ax.set_title(title, pad=6)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Fraction of area')
                ax.set_yticklabels([])
                sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
                continue

            pivot = (df_ml
                     .pivot_table(index='biome_group', columns='cls', values='frac', aggfunc='mean')
                     .reindex(groups_plot))
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

        # one legend on top
        handles = [mpl.patches.Patch(color=colors[c], label=c) for c in classes]
        fig.legend(handles=handles, ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(f"{var_name}: Biome confusion — {fig_tag} — {method}", y=1.18, fontsize=13)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"{var_name}_{method}_biome_confusion_{fig_tag}_row.svg")
        fig.savefig(fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)


# --- Maps ---


def plot_map_indicators(ds_breaks, ds_ews, var_name, out_dir):
    """Per-indicator class maps and bars vs each CP method."""
    alpha = 0.05
    indicators = [f'{var_name}_ac1', f'{var_name}_std', f'{var_name}_fd',
                  f'{var_name}_skew', f'{var_name}_kurt']

    methods = {
        'pettitt':  {'cp': ds_breaks['pettitt_cp'],     'pval': ds_breaks['pettitt_pval']},
        'stc':      {'cp': ds_breaks['strucchange_bp'], 'pval': ds_breaks['Fstat_pval']},
        'variance': {'cp': ds_breaks['bp_var'],         'pval': ds_breaks['pval_var']},
    }

    lon_name, lat_name = detect_lon_lat_names(ds_ews)
    land_mask = make_land_mask(ds_ews, lon_name, lat_name)

    def ews_mask(ds, var):
        tau = ds[f'{var}_kt']; p = ds[f'{var}_pval']
        base = (p < alpha) & tau.notnull()
        if var in [f'{var_name}_skew', f'{var_name}_kurt']:
            base &= (tau > 0)
        if var == f'{var_name}_fd':
            base &= (tau < 0)
        return base.where(land_mask, False).fillna(False)

    def break_mask(method):
        cp = methods[method]['cp']; p = methods[method]['pval']
        return ((p < alpha) & (cp > 0)).where(land_mask, False).fillna(False)

    for v in indicators:
        print(f"Mapping indicator: {v}")
        valid = land_mask & ds_ews[f'{v}_kt'].notnull()
        e_mask = ews_mask(ds_ews, v).where(valid, False)

        for method in methods:
            b_mask = break_mask(method).where(valid, False)
            TP, FP, FN, TN = compute_confusion_arrays(e_mask, b_mask)
            cmap = class_map_from_confusion(TP, FP, FN, TN).where(valid)

            label = v
            slabel = safe_label(label)
            fig_map = plot_class_map(cmap, f"{label} — {method}")
            fig_map.savefig(os.path.join(out_dir, f"{slabel}_{method}_map.png"),
                            format='png', dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig_map)

            fig_bar = plot_class_bar(cmap, f"{label} — {method}")
            fig_bar.savefig(os.path.join(out_dir, f"{slabel}_{method}_bar.svg"),
                            format='svg', dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig_bar)


def plot_map_ews(ds_breaks, ds_ews, var_name, out_dir):
    """Per-composite EWS class maps and bars vs each CP method."""
    alpha = 0.05
    methods = {
        'pettitt':  {'cp': ds_breaks['pettitt_cp'],     'pval': ds_breaks['pettitt_pval']},
        'stc':      {'cp': ds_breaks['strucchange_bp'], 'pval': ds_breaks['Fstat_pval']},
        'variance': {'cp': ds_breaks['bp_var'],         'pval': ds_breaks['pval_var']},
    }

    lon_name, lat_name = detect_lon_lat_names(ds_ews)
    land_mask = make_land_mask(ds_ews, lon_name, lat_name)

    def ews_dir_mask(ds, var, direction):
        tau = ds[f'{var}_kt']; p = ds[f'{var}_pval']
        m = (p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)
        return m.where(land_mask, False).fillna(False)

    composites = {
        'ac1_std_inc': ('AC1 up & SD up',
            lambda ds: ews_dir_mask(ds, f'{var_name}_ac1', 'inc') & ews_dir_mask(ds, f'{var_name}_std', 'inc')),
        'ac1_std_dec': ('AC1 down & SD down',
            lambda ds: ews_dir_mask(ds, f'{var_name}_ac1', 'dec') & ews_dir_mask(ds, f'{var_name}_std', 'dec')),
        'ac1_std_mixed': ('AC1 updown & SD updown', lambda ds: (
            (ews_dir_mask(ds, f'{var_name}_ac1', 'inc') & ews_dir_mask(ds, f'{var_name}_std', 'dec')) |
            (ews_dir_mask(ds, f'{var_name}_ac1', 'dec') & ews_dir_mask(ds, f'{var_name}_std', 'inc'))
        )),
        'skew_kurt_inc': ('Skew up & Kurt up',
            lambda ds: ews_dir_mask(ds, f'{var_name}_skew', 'inc') & ews_dir_mask(ds, f'{var_name}_kurt', 'inc')),
        'fd_dec': ('FD down', lambda ds: ews_dir_mask(ds, f'{var_name}_fd', 'dec')),
    }

    def break_mask(method):
        cp = methods[method]['cp']; p = methods[method]['pval']
        return ((p < alpha) & (cp > 0)).where(land_mask, False).fillna(False)

    for key, (label, mask_fn) in composites.items():
        print(f"Mapping EWS : {label}")

        if key.startswith('ac1_std'):
            needed = [f'{var_name}_ac1', f'{var_name}_std']
        elif key == 'skew_kurt_inc':
            needed = [f'{var_name}_skew', f'{var_name}_kurt']
        else:
            needed = [f'{var_name}_fd']

        valid = land_mask.copy()
        for v in needed:
            valid &= ds_ews[f'{v}_kt'].notnull()

        e_mask = mask_fn(ds_ews).where(valid, False)
        for method in methods:
            b_mask = break_mask(method).where(valid, False)
            TP, FP, FN, TN = compute_confusion_arrays(e_mask, b_mask)
            cmap = class_map_from_confusion(TP, FP, FN, TN).where(valid)

            slabel = safe_label(label)
            fig_map = plot_class_map(cmap, f"{label} — {method}")
            fig_map.savefig(os.path.join(out_dir, f"{slabel}_{method}_map.png"),
                            format='png', dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig_map)

            fig_bar = plot_class_bar(cmap, f"{label} — {method}")
            fig_bar.savefig(os.path.join(out_dir, f"{slabel}_{method}_bar.svg"),
                            format='svg', dpi=300, facecolor='white', bbox_inches='tight')
            plt.close(fig_bar)


# --- same bar plots but aridity class instead of biome --- 

def get_area_da(lat_vals, lon_vals):
    """Area (km2) per grid cell """
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
  
    # ERA5 precip monthly (m) → mm
    precip = xr.open_dataset("/mnt/data/romi/data/ERA5_0.25_monthly/total_precipitation/total_precipitation_monthly.nc")\
              .sel(time=slice("2000-01-01", "2023-12-31"))\
              .rename({"latitude": "lat", "longitude": "lon"})
    precip = precip.rio.write_crs("EPSG:4326")
    precip = wrap_to_180(precip, "lon")
    precip.rio.set_spatial_dims("lon", "lat", inplace=True)
    precip = precip * 1000.0  # m → mm (per month)

    # PET monthly (already monthly mean/sum, ensure monthly frequency)
    pet = xr.open_dataset("/mnt/data/romi/data/et_pot/monthly_sum_epot_clean.zarr")\
             .sel(time=slice("2000-01-01", "2023-11-30"))

    # Align monthly timestamps
    precip["time"] = ("time", pd.date_range(str(precip.time.values[0])[:10], periods=precip.sizes["time"], freq="MS"))
    pet   ["time"] = ("time", pd.date_range(str(pet.time.values[0])[:10],     periods=pet.sizes["time"],     freq="MS"))

    # Avoid zeros
    precip_tp = precip["tp"].where(precip["tp"] != 0, 1e-6)
    pet_pet   = pet["pet"].where(pet["pet"] != 0, 1e-6)

    # Land-sea mask for precip grid, then stick to land only
    lsm = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")\
            .rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = wrap_to_180(lsm, "lon")
    lsm.rio.set_spatial_dims("lon", "lat", inplace=True)
    lsm_on_precip = lsm["lsm"].interp(lat=precip.lat, lon=precip.lon)

    precip_tp = precip_tp.where(lsm_on_precip > 0.5)

    # Align & compute aridity index (precip/PET), time-mean
    precip_aligned, pet_aligned = xr.align(precip_tp, pet_pet, join="inner")
    aridity_index = (precip_aligned / pet_aligned).rename("aridity_index").where(lsm_on_precip > 0.5)
    aridity_mean  = aridity_index.mean(dim="time")

    # Interp to analysis grid
    lonn, latt = detect_lon_lat_names(ds_target)
    aridity_on_target = aridity_mean.interp(lat=ds_target[latt], lon=ds_target[lonn])

    # Bin and map to labels
    class_index = xr.apply_ufunc(
        np.digitize, aridity_on_target,
        kwargs={"bins": ARIDITY_BINS},
        vectorize=True, dask="parallelized",
        output_dtypes=[int],
    )

    def idx_to_label(idx):
        if 1 <= idx <= len(ARIDITY_LABELS):
            return ARIDITY_LABELS[idx - 1]
        return np.nan

    label_array = xr.apply_ufunc(
        np.vectorize(idx_to_label),
        class_index,
        vectorize=True, dask="parallelized",
        output_dtypes=[object],
    ).rename("aridity_class")

    # Ensure coords names match ds_target 'lat','lon' for area calc later
    if lonn != "lon" or latt != "lat":
        label_array = label_array.rename({lonn: "lon", latt: "lat"})
    return label_array

def _indicator_short_name(full_var: str) -> str:
    suffix = full_var.split("_")[-1]
    lookup = dict(INDICATORS_META)
    return lookup.get(suffix, suffix.upper())

def _aridity_confusion_fraction_table(ds_breaks, ds_ews, var_name, which="indicators"):
  
    alpha, indicators, _methods = _global_masks_and_methods(ds_breaks, ds_ews, var_name)

    # Valid 
    urban = xr.open_dataset("/mnt/data/romi/data/urban_mask.zarr").rio.write_crs("EPSG:4326").interp_like(ds_ews, method="nearest")
    urban = urban["urban-coverfraction"].squeeze("time", drop=True)
    lsm   = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")\
              .rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = wrap_to_180(lsm, "lon")
    lsm = lsm.rio.set_spatial_dims("lon", "lat", inplace=True) or lsm
    lsm_on_ds = lsm["lsm"].interp(lat=ds_ews["lat"], lon=ds_ews["lon"])

    valid_mask = ((lsm_on_ds > 0.5) | lsm_on_ds.isnull()) & ((urban <= 3) | urban.isnull())

    # Aridity classes on ds_ews grid (strings) and area weights
    aridity_class = compute_aridity_classes_like(ds_ews).where(valid_mask)
    area_da = get_area_da(ds_ews["lat"].values, ds_ews["lon"].values).where(valid_mask)

    # Methods
    methods = {
        'pettitt':  {'cp': ds_breaks['pettitt_cp'],     'pval': ds_breaks['pettitt_pval']},
        'stc':      {'cp': ds_breaks['strucchange_bp'], 'pval': ds_breaks['Fstat_pval']},
        'variance': {'cp': ds_breaks['bp_var'],         'pval': ds_breaks['pval_var']},
    }

    # Local helpers on the full grid; we mask per class later
    def ews_dir_mask(var, direction):
        tau = ds_ews[f'{var}_kt']; p = ds_ews[f'{var}_pval']
        return ((p < alpha) & (tau > 0 if direction == 'inc' else tau < 0)).fillna(False)

    composites = {
        'AC1 up & SD up':    lambda: ews_dir_mask(f'{var_name}_ac1','inc') & ews_dir_mask(f'{var_name}_std','inc'),
        'AC1 down & SD down':    lambda: ews_dir_mask(f'{var_name}_ac1','dec') & ews_dir_mask(f'{var_name}_std','dec'),
        'AC1 updown & SD updown':  lambda: (ews_dir_mask(f'{var_name}_ac1','inc') & ews_dir_mask(f'{var_name}_std','dec')) |
                                   (ews_dir_mask(f'{var_name}_ac1','dec') & ews_dir_mask(f'{var_name}_std','inc')),
        'Skew up & Kurt up': lambda: ews_dir_mask(f'{var_name}_skew','inc') & ews_dir_mask(f'{var_name}_kurt','inc'),
        'FD down':            lambda: ews_dir_mask(f'{var_name}_fd','dec'),
    }

    rows = []
    if which == "indicators":
        label_iter = [f'{var_name}_{k}' for k, _ in INDICATORS_META]
    else:
        label_iter = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']

    for label in label_iter:
        # Indicator mask
        if which == "indicators":
            tau = ds_ews[f'{label}_kt']; p = ds_ews[f'{label}_pval']
            base = (p < alpha) & tau.notnull()
            if label.endswith("_skew") or label.endswith("_kurt"):
                base &= (tau > 0)
            if label.endswith("_fd"):
                base &= (tau < 0)
            e_mask = base.fillna(False)
        else:
            e_mask = composites[label]().fillna(False)

        for method, mpair in methods.items():
            cp = mpair['cp']; pv = mpair['pval']
            b_mask = ((pv < alpha) & (cp > 0)).fillna(False)

            # Per aridity class, area-weighted fractions
            for cls_label in ARIDITY_PLOT:
                cls_mask = (aridity_class == cls_label)
                if not bool(cls_mask.any()):
                    continue
                area_cls = area_da.where(cls_mask)
                if float(area_cls.sum().values) == 0.0:
                    continue

                TP, FP, FN, TN = compute_confusion_arrays(e_mask & cls_mask, b_mask & cls_mask)
                TP_a = float(area_cls.where(TP).sum().values)
                FP_a = float(area_cls.where(FP).sum().values)
                FN_a = float(area_cls.where(FN).sum().values)
                TN_a = float(area_cls.where(TN).sum().values)
                tot  = TP_a + FP_a + FN_a + TN_a
                if tot <= 0:
                    continue
                rows.extend([
                    {"aridity_class": cls_label, "method": method, "label": label, "cls": "TP", "frac": TP_a / tot},
                    {"aridity_class": cls_label, "method": method, "label": label, "cls": "FP", "frac": FP_a / tot},
                    {"aridity_class": cls_label, "method": method, "label": label, "cls": "FN", "frac": FN_a / tot},
                    {"aridity_class": cls_label, "method": method, "label": label, "cls": "TN", "frac": TN_a / tot},
                ])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["aridity_class"] = pd.Categorical(df["aridity_class"], categories=ARIDITY_PLOT, ordered=True)
    return df.sort_values(["method", "label", "aridity_class", "cls"])

def plot_aridity_confusion_rows(df_long, var_name, out_dir, which="indicators"):
   
    colors = {'TP': '#7d475dff', 'FP': '#ffeaefff', 'FN': '#e9e2bcff', 'TN': '#2c787eff'}
    classes = ['TP', 'FP', 'FN', 'TN']

    methods = ['pettitt', 'stc', 'variance']
    if which == "indicators":
        labels_order = [f"{var_name}_{k}" for k, _ in INDICATORS_META]
        titles       = [_indicator_short_name(lab) for lab in labels_order]
        fig_tag      = "indicators"
    else:
        labels_order = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']
        titles       = labels_order
        fig_tag      = "ews"

    for method in methods:
        df_m = df_long[df_long['method'] == method]
        if df_m.empty:
            continue

        sn.set_style("whitegrid")
        fig, axes = plt.subplots(1, 5, figsize=(3.9*5, 3.6), sharey=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, label, title in zip(axes, labels_order, titles):
            df_ml = df_m[df_m['label'] == label]
            if df_ml.empty:
                ax.set_title(title, pad=6)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Fraction of area')
                ax.set_yticklabels([])
                sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
                continue

            pivot = (df_ml
                     .pivot_table(index='aridity_class', columns='cls', values='frac', aggfunc='mean')
                     .reindex(ARIDITY_PLOT))
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

        # one legend on top
        handles = [mpl.patches.Patch(color=colors[c], label=c) for c in classes]
        fig.legend(handles=handles, ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(f"{var_name}: Aridity confusion — {fig_tag} — {method}", y=1.18, fontsize=13)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"{var_name}_{method}_aridity_confusion_{fig_tag}_row.svg")
        fig.savefig(fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)

# NEW: Heatmaps of the F1 scores for aridity and biomes 

def _f1_from_fracs(tp, fp, fn):
    denom = 2*tp + fp + fn
    return (2*tp / denom) if denom > 0 else np.nan

def metric_scores(TP, FP, FN, TN):
    denom_f1 = (2*TP + FP + FN)
    f1 = (2*TP) / denom_f1 if denom_f1 > 0 else np.nan
    prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    rec  = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    acc  = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else np.nan
    bal_acc = 0.5*((rec if not np.isnan(rec) else 0) + (spec if not np.isnan(spec) else 0))
    mcc_num = (TP*TN - FP*FN)
    mcc_den = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = mcc_num/mcc_den if mcc_den > 0 else np.nan
    return {
        "F1": f1, "Precision": prec, "Recall": rec, "Specificity": spec,
        "Accuracy": acc, "BalancedAccuracy": bal_acc, "MCC": mcc
    }


def _heatmap_from_long_f1(df_long, index_col, index_order, labels_order, title, out_fp):
    """
    df_long: columns = [<index_col>, 'label', 'cls', 'frac'] filtered to one method ('stc') and composites only.
    Builds an (index_order x labels_order) matrix of Balanced Accuracy using area-weighted fractions per group.
    """
    def balanced_accuracy(tp, fp, fn, tn):
        # Recall (TPR) and Specificity (TNR)
        pos_denom = tp + fn
        neg_denom = tn + fp
        if pos_denom == 0 or neg_denom == 0:
            return np.nan  # undefined if no positives or no negatives
        rec  = tp / pos_denom
        spec = tn / neg_denom
        return 0.5 * (rec + spec)

    recs = []
    supports = []  # optional: total "defined" support (positives + negatives) for later weighted summaries
    for idx in index_order:
        df_i = df_long[df_long[index_col] == idx]
        for lab in labels_order:
            df_il = df_i[df_i['label'] == lab]
            if df_il.empty:
                recs.append((idx, lab, np.nan))
                supports.append((idx, lab, 0.0))
                continue

            tp = float(df_il[df_il['cls'] == 'TP']['frac'].sum())
            fp = float(df_il[df_il['cls'] == 'FP']['frac'].sum())
            fn = float(df_il[df_il['cls'] == 'FN']['frac'].sum())
            tn = float(df_il[df_il['cls'] == 'TN']['frac'].sum())

            # support for BA: positives + negatives that make BA defined
            pos = tp + fn
            neg = tn + fp
            if pos == 0 or neg == 0:
                recs.append((idx, lab, np.nan))   # undefined → blank cell
                supports.append((idx, lab, 0.0))
                continue

            ba = balanced_accuracy(tp, fp, fn, tn)
            recs.append((idx, lab, ba))
            supports.append((idx, lab, pos + neg))

    mat = (pd.DataFrame(recs, columns=[index_col, 'label', 'BA'])
              .pivot(index=index_col, columns='label', values='BA')
              .reindex(index_order)[labels_order])

    
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "custom_div",
        ["#b97474ff", "white", "#85b6c5ff"]   # red → white → blue
    )

    norm = TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=0.6)

    annot = mat.copy()
    mask = mat.isna()
    annot_str = annot.applymap(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(1.6*len(labels_order), 0.5*len(index_order) + 2))


    sn.heatmap(mat, mask=mask, vmin=0.4, vmax=0.6, cmap=cmap, norm = norm, # Et: 0.4 to 0.6, 
               annot=annot_str, fmt="", annot_kws={"fontsize": 9},
               cbar_kws={'label': 'Balanced accuracy'}, linewidths=0.3, linecolor='white', ax=ax)

    ax.set_xlabel('Composite EWS')
    ax.set_ylabel(index_col.replace('_', ' ').title())
    ax.set_title(title, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    fig.savefig(out_fp, format='svg', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_f1_heatmap_biomes_stc(ds_breaks, ds_ews, var_name, out_dir):
    """
    Heatmap of F1 by biome group (rows) vs composite EWS (cols), for method='stc' only.
    Uses area-weighted fractions from _biome_confusion_fraction_table.
    """
    df_bio = _biome_confusion_fraction_table(ds_breaks=ds_breaks, ds_ews=ds_ews,
                                             var_name=var_name, which="composites")
    if df_bio.empty:
        return
    df_bio = df_bio[df_bio['method'] == 'stc'].copy()

    labels_order = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']
    title = f"{var_name}: F1 by biome (composites, stc)"
    out_fp = os.path.join(out_dir, f"{var_name}_stc_biome_f1_heatmap_ews.svg")
    _heatmap_from_long_f1(df_bio, index_col='biome_group', index_order=groups_plot,
                          labels_order=labels_order, title=title, out_fp=out_fp)

def plot_f1_heatmap_aridity_stc(ds_breaks, ds_ews, var_name, out_dir):
    """
    Heatmap of F1 by aridity class (rows) vs composite EWS (cols), for method='stc' only.
    Uses area-weighted fractions from _aridity_confusion_fraction_table.
    """
    df_arid = _aridity_confusion_fraction_table(ds_breaks=ds_breaks, ds_ews=ds_ews,
                                                var_name=var_name, which="composites")
    if df_arid.empty:
        return
    df_arid = df_arid[df_arid['method'] == 'stc'].copy()

    labels_order = ['AC1 up & SD up', 'AC1 down & SD down', 'AC1 updown & SD updown', 'Skew up & Kurt up', 'FD down']
    title = f"{var_name}: F1 by aridity class (composites, stc)"
    out_fp = os.path.join(out_dir, f"{var_name}_stc_aridity_f1_heatmap_ews.svg")
    _heatmap_from_long_f1(df_arid, index_col='aridity_class', index_order=ARIDITY_PLOT,
                          labels_order=labels_order, title=title, out_fp=out_fp)

# --- Main ---

def main():
    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    p = argparse.ArgumentParser(
        description='Plot & map indicator/EWS performance with changepoints as ground truth.'
    )
    p.add_argument('--ews_kt_path', type=str, required=True, help='Path to the EWS Kendall tau dataset.')
    p.add_argument('--ds_cp_path', type=str, required=True, help='Path to the changepoint dataset.')
    p.add_argument('--var', type=str, required=True, help='Variable name (e.g., Et, precip, sm).')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory for figures.')
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    ews_kt_path = args.ews_kt_path
    ds_cp_path  = args.ds_cp_path
    var_name    = args.var
    out_dir     = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load
    ds_ews = xr.open_dataset(ews_kt_path)
    ds_cp  = xr.open_dataset(ds_cp_path)

    # Detect coords and apply precip land mask 
    lon_name, lat_name = detect_lon_lat_names(ds_ews)
    ds_cp, ds_ews = apply_land_mask_if_precip(ds_cp, ds_ews, var_name, lon_name, lat_name)

    # 1) PERFORMANCE SCORES (global)
    print('--- Global metrics: Indicators (F1 / Precision / Accuracy) ---')
    for metric in ["F1", "Precision", "Accuracy"]:
        plot_metric_indicators(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir, metric=metric)

    print('--- Global metrics: EWS composites (F1 / Precision / Accuracy) ---')
    for metric in ["F1", "Precision", "Accuracy"]:
        plot_metric_ews(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir, metric=metric)

    # 2) MAPS + GLOBAL BARS 
    print('--- Mapping per-indicator performance (maps + global bars) ---')
    plot_map_indicators(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir)

    print('--- Mapping composite EWS performance (maps + global bars) ---')
    plot_map_ews(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir)

    # 3) 5-PANEL BIOME-BASED BAR PLOTS
    print('--- Biome confusion fractions (for 5-panel figures): indicators ---')
    df_bio_conf_ind = _biome_confusion_fraction_table(ds_breaks=ds_cp, ds_ews=ds_ews,
                                                      var_name=var_name, which="indicators")
    plot_biome_confusion_rows(df_bio_conf_ind, var_name, out_dir, which="indicators")

    print('--- Biome confusion fractions (for 5-panel figures): composites ---')
    df_bio_conf_comp = _biome_confusion_fraction_table(ds_breaks=ds_cp, ds_ews=ds_ews,
                                                       var_name=var_name, which="composites")
    plot_biome_confusion_rows(df_bio_conf_comp, var_name, out_dir, which="composites")

    # 4) Aridity !
    print('--- Aridity confusion fractions (5-panel): indicators ---')
    df_arid_ind = _aridity_confusion_fraction_table(ds_breaks=ds_cp, ds_ews=ds_ews,
                                                    var_name=var_name, which="indicators")
    plot_aridity_confusion_rows(df_arid_ind, var_name, out_dir, which="indicators")

    print('--- Aridity confusion fractions (5-panel): composites ---')
    df_arid_comp = _aridity_confusion_fraction_table(ds_breaks=ds_cp, ds_ews=ds_ews,
                                                     var_name=var_name, which="composites")
    plot_aridity_confusion_rows(df_arid_comp, var_name, out_dir, which="composites")

    # 5) F1 heatmaps only for stc and only the composite indicators 
    print('--- F1 heatmap by biome (composites, stc) ---')
    plot_f1_heatmap_biomes_stc(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir)

    print('--- F1 heatmap by aridity class (composites, stc) ---')
    plot_f1_heatmap_aridity_stc(ds_breaks=ds_cp, ds_ews=ds_ews, var_name=var_name, out_dir=out_dir)


    print('--- Done ---')

if __name__ == '__main__':
    main()