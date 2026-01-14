#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import rioxarray  
import matplotlib.colors as mcolors
from pathlib import Path    
from collections import OrderedDict
from utils.config import load_config, cfg_path, cfg_get

""" 
Plots bar graphs of the percentage of land area of increasing or decreasing indicators. 
Biomes have been grouped for brevity. 

E.g. 
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr --var 'Et' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr --var 'sm' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr --var 'precip' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
 
"""

# ----- shared colours & sizes to match kt map script) -----

MM_TO_IN = 1.0 / 25.4

FIGSIZE_COMBINED = (
    100.327 * MM_TO_IN,  # width in inches
    40.0 * MM_TO_IN      # height in inches
)

PINK = "#82315E"   
GREEN = "#256D15" 

PINK_SOFT = sn.desaturate(PINK, 0.95)
GREEN_SOFT = sn.desaturate(GREEN, 0.95)

NEUTRAL_COLOR = "#c4c4c4"  # for neutral

# mixed = overlap of pink & green in RGB space (edit in affinity)
_mixed_rgb = (
    np.array(mcolors.to_rgb(PINK_SOFT)) +
    np.array(mcolors.to_rgb(GREEN_SOFT))
) / 2.0
MIXED_COLOR = mcolors.to_hex(_mixed_rgb)

# colours for indicator mode (Decrease/Neutral/Increase)
INDICATOR_COLORS = {
    "Decrease": PINK_SOFT,
    "Neutral":  NEUTRAL_COLOR,
    "Increase": GREEN_SOFT,
}



# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--var", required=True, help="sm, Et, precip")
    p.add_argument("--suffix", default=None,
                   help="Optional suffix for inferred dataset (e.g. breakpoint_stc).")
    p.add_argument("--dataset", default=None,
                   help="Optional override path to *_kt.zarr. If omitted, inferred from config + --var + --suffix.")
    p.add_argument("--outdir", default=None,
                   help="Optional override output directory. If omitted, uses outputs_root/figures/biomes/<var>/")
    p.add_argument("--mode", choices=["indicators", "combined"], default="indicators")
    p.add_argument("--config", default="config.yaml")

    return p.parse_args()

# --- Data & helpers ---

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

indicators = [
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



def combined_family_masks(ds, prefix):
    """
    Families of combined signals with both directions in the same family:
      - AC1 & SD: Both up, Mixed (one up one down), Both down
      - Fractal dimension: FD up, FD down
      - Flickering: Skew up & Kurt up, Skew down & Kurt down
    All must be significant at p < 0.05.
    """
    ac1  = ds[f"{prefix}_ac1_kt"];  ac1_p  = ds[f"{prefix}_ac1_pval"]
    std  = ds[f"{prefix}_std_kt"];  std_p  = ds[f"{prefix}_std_pval"]
    fd   = ds[f"{prefix}_fd_kt"];   fd_p   = ds[f"{prefix}_fd_pval"]
    skew = ds[f"{prefix}_skew_kt"]; skew_p = ds[f"{prefix}_skew_pval"]
    kurt = ds[f"{prefix}_kurt_kt"]; kurt_p = ds[f"{prefix}_kurt_pval"]

    sig = lambda p: (p < 0.05)

    # AC1 & SD
    both_up  = (sig(ac1_p) & (ac1 > 0)) & (sig(std_p) & (std > 0))
    both_down= (sig(ac1_p) & (ac1 < 0)) & (sig(std_p) & (std < 0))
    mix1     = (sig(ac1_p) & (ac1 > 0)) & (sig(std_p) & (std < 0))
    mix2     = (sig(ac1_p) & (ac1 < 0)) & (sig(std_p) & (std > 0))
    mixed    = (mix1 | mix2)

    # FD
    fd_up    = (sig(fd_p) & (fd > 0))
    fd_down  = (sig(fd_p) & (fd < 0))

    # Flickering
    flick_up     = (sig(skew_p) & (skew > 0)) & (sig(kurt_p) & (kurt > 0))
    flick_down   = (sig(skew_p) & (skew < 0)) & (sig(kurt_p) & (kurt < 0))
    flick_mix_1  = (sig(skew_p) & (skew > 0)) & (sig(kurt_p) & (kurt < 0))
    flick_mix_2  = (sig(skew_p) & (skew < 0)) & (sig(kurt_p) & (kurt > 0))
    flick_mixed  = (flick_mix_1 | flick_mix_2)

    return OrderedDict([
        ("AC1 & SD", OrderedDict([
            ("Both up (AC1 up & SD up)", both_up),
            ("Mixed (one up, one down)", mixed),
            ("Both down (AC1 down & SD down)", both_down),
        ])),
        ("Fractal dimension", OrderedDict([
            ("FD up", fd_up),
            ("FD down", fd_down),
        ])),
        ("Flickering (Skew & Kurt)", OrderedDict([
            ("Skew up & Kurt up", flick_up),
            ("Mixed (one up, one down)", flick_mixed),
            ("Skew down & Kurt down", flick_down),
        ])),
    ])


def get_biomes(biome: str, tnc_shp: str):
    data = gpd.read_file(tnc_shp, where=f"WWF_MHTNAM='{biome}'")
    unique_ids = [i for i in data.WWF_MHTNAM.unique()]
    unique_num = [i for i in data.WWF_MHTNUM.unique()]
    biomes_df = pd.DataFrame(unique_ids)
    biomes_df["label"] = unique_num
    pd.set_option("display.max_colwidth", None)
    return data, biomes_df

def clip_biomes(ds, gdf):
    ds = ds.rio.write_crs('EPSG:4326')
    ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    clipped = ds.rio.set_crs('WGS84').rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped

def clip_da(da, gdf):
    """Clip a DataArray with the same rioxarray path as datasets."""
    da = da.rio.write_crs('EPSG:4326')
    da.rio.set_spatial_dims("lon", "lat", inplace=True)
    return da.rio.set_crs('WGS84').rio.clip(gdf.geometry, gdf.crs, drop=True)


def get_area_grid(lat, dlon=0.25, dlat=0.25):
    """1D lat band area (km2) for each latitude, matching your notebook."""
    R = 6371  # km
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)

def area_da_for(clipped_ds):
    """Broadcast that 1D area to 2D"""
    latitudes = clipped_ds['lat'].values
    area_per_lat = get_area_grid(latitudes)
    area_grid = np.broadcast_to(area_per_lat[:, np.newaxis], (latitudes.size, clipped_ds['lon'].size))
    return xr.DataArray(area_grid, coords={'lat': latitudes, 'lon': clipped_ds['lon'].values}, dims=('lat', 'lon'))

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


def combined_signal_masks(ds, prefix):
    """
    Return dict[str -> xr.DataArray(bool)] for:
      - CSD: AC1 up & SD up
      - CSU: AC1 down & SD down
      - Mixed: (AC1 up & SD down) | (AC1 down & SD up)
      - FD: FD down
      - Flickering: Skew up & Kurt up
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
        "CSD (AC1 up & SD up)": CSD,
        "CSU (AC1 down & SD down)": CSU,
        "Mixed (AC1 updown & SD updown)": Mixed,
        "FD down": FDdecline,
        "Flickering (Skew up & Kurt up)": Flicker,
    }

# ---------------- Main ----------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)

    def _sfx(s):
        if not s:
            return ""
        s = str(s).strip()
        return s if s.startswith("_") else f"_{s}"

    default_ds = Path(outputs_root) / "zarr" / f"out_{args.var}{_sfx(args.suffix)}_kt.zarr"
    ds_path = Path(args.dataset) if args.dataset else default_ds

    default_outdir = Path(outputs_root) / "figures" / "biomes" / args.var
    outdir = Path(args.outdir) if args.outdir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)



    TNC_SHP = str(cfg_path(cfg, "resources.tnc_biomes_shapefile", must_exist=True))
    

    # dataset + masks (
    ds = xr.open_dataset(str(ds_path))
    urban_mask = xr.open_dataset(cfg_path(cfg, "resources.urban_mask_zarr", must_exist=True))
    crop_mask  = xr.open_dataset(cfg_path(cfg, "resources.crop_mask_zarr", must_exist=True))

    urban_mask = urban_mask.rio.write_crs("EPSG:4326")
    crop_mask  = crop_mask.rio.write_crs("EPSG:4326")

    urban_mask = urban_mask.interp_like(ds, method="nearest")
    crop_mask  = crop_mask.interp_like(ds, method="nearest")

    urban_mask = urban_mask['urban-coverfraction'].squeeze('time')
    crop_mask  = crop_mask['crops-coverfraction'].squeeze('time')

    prefix = args.var  # e.g., "sm", "Et", "precip"

    IRRIG_PATH = cfg_path(cfg, "resources.irrigated_areas", must_exist=True)
    IRRIG_THRESH_PERCENT = 60.0  # AEI > 60% = considered irrigated and excluded from cropland biome

    irrig_ds = xr.open_dataset(IRRIG_PATH).rio.write_crs("EPSG:4326").interp_like(ds, method="nearest")
    if len(irrig_ds.data_vars) == 0:
        raise ValueError("Irrigation dataset has no data variables.")
    _irrig_da = list(irrig_ds.data_vars.values())[0].squeeze()
    try:
        _vmax = float(_irrig_da.max().values)
    except Exception:
        _vmax = 1.0
    irrig_frac = _irrig_da / 100.0 if _vmax > 1.5 else _irrig_da

    # base cropland mask 
    crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())

    # rain-fed cropland (what we will use for the cropland biome)
    irrigated = (irrig_frac >= IRRIG_THRESH_PERCENT / 100.0)
    rainfed_cropland = crop_only & (~irrigated | irrigated.isnull())
        
    # ---------- INDIVIDUAL INDICATORS  ----------

    if args.mode == "indicators":
       
        group_totals = {k: {g: {"dec":0.0,"neu":0.0,"inc":0.0,"tot":0.0} for g in groups_plot} for k,_ in indicators}

        # natural biomes to grouped
        for biome in BIOMES:
            if biome == "Mangroves":
                continue
            grp = GROUP_MAP.get(biome)
            if grp is None:
                continue

            gdf, _ = get_biomes(biome, tnc_shp=str(TNC_SHP))
            clipped = clip_biomes(ds, gdf)

            valid_biome_mask = ((urban_mask <= 3) | urban_mask.isnull()) & (~crop_only)
            clipped = clipped.where(valid_biome_mask)

            area_da = area_da_for(clipped)

            for key, _label in indicators:
                kt   = clipped[f"{prefix}_{key}_kt"]
                pval = clipped[f"{prefix}_{key}_pval"]
                dec_a, neu_a, inc_a, tot_a = tri_split_areas(kt, pval, area_da)
                acc = group_totals[key][grp]
                acc["dec"] += dec_a; acc["neu"] += neu_a; acc["inc"] += inc_a; acc["tot"] += tot_a

        # pseudo-biome: cropland (not rainfed only)
        grp = "Cropland"
        crop_clipped = ds.where(rainfed_cropland)
        area_da = area_da_for(crop_clipped)
        for key, _label in indicators:
            kt   = crop_clipped[f"{prefix}_{key}_kt"]
            pval = crop_clipped[f"{prefix}_{key}_pval"]
            dec_a, neu_a, inc_a, tot_a = tri_split_areas(kt, pval, area_da)
            acc = group_totals[key][grp]
            acc["dec"] += dec_a; acc["neu"] += neu_a; acc["inc"] += inc_a; acc["tot"] += tot_a

        # build dfs
        dfs = {}
        for key, label in indicators:
            rows = []
            for g in groups_plot:
                tot = group_totals[key][g]["tot"]
                if tot > 0:
                    dec = 100.0 * group_totals[key][g]["dec"] / tot
                    neu = 100.0 * group_totals[key][g]["neu"] / tot
                    inc = 100.0 * group_totals[key][g]["inc"] / tot
                    rows.append({"Group": g, "Decrease": dec, "Neutral": neu, "Increase": inc})

            df = pd.DataFrame(rows) if rows else pd.DataFrame({
                "Group": groups_plot,
                "Decrease": np.zeros(len(groups_plot)),
                "Neutral":  np.zeros(len(groups_plot)),
                "Increase": np.zeros(len(groups_plot)),
            })

            df["Group"] = pd.Categorical(df["Group"], categories=groups_plot, ordered=True)
            dfs[key] = df.sort_values("Group")

        # plot
        sn.set_style("whitegrid")
        fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

        colors = {
            "Decrease": "#64afc9ff",  # blue
            "Neutral":  "#bab9b9",    # light grey
            "Increase": "#cd5d5dff",  # red
        }

        for ax, (key, label) in zip(axes, indicators):
            dfp = dfs[key].copy()
            left = np.zeros(len(dfp))
            for col in ["Decrease", "Neutral", "Increase"]:
                ax.barh(
                    y=dfp["Group"],
                    width=dfp[col],
                    left=left,
                    color=colors[col],
                    edgecolor="none",
                    height=0.87,
                )
                left += dfp[col].to_numpy()

            ax.set_title(label, fontsize=14, pad=8)
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of group area")
            ax.grid(False)

        axes[0].invert_yaxis()
        for ax in axes:
            ax.tick_params(axis='y', labelsize=12)
            sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
            ax.margins(y=0.02)

        plt.tight_layout()
        out_svg = os.path.join(str(outdir), f"{prefix}_kt_biomes.svg")
        fig.savefig(out_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)
        


        return


    # ---------- COMBINED INDICATORS  ----------

    families = combined_family_masks(ds, prefix)

    # Land-sea mask for denominators
    mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
    lsm = xr.open_dataset(mask_path, engine="cfgrib")
    lsm = lsm.rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = lsm.assign_coords(lon=((lsm.lon + 180) % 360) - 180).sortby("lon")
    lsm = lsm.interp_like(ds, method="nearest")
    land = (lsm["lsm"] > 0.5) | lsm["lsm"].isnull()

    # Area grid 
    lat = ds["lat"].values
    lon = ds["lon"].values
    dlat = float(np.abs(np.diff(lat).mean())) if lat.size > 1 else 0.25
    dlon = float(np.abs(np.diff(lon).mean())) if lon.size > 1 else 0.25

    def area_da_custom(lat_vals, lon_vals):
        R = 6371.0
        dlat_rad = np.radians(dlat)
        dlon_rad = np.radians(dlon)
        lat_rad = np.radians(lat_vals)
        area_row = (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
        grid = np.broadcast_to(area_row[:, None], (lat_vals.size, lon_vals.size))
        return xr.DataArray(grid, coords={"lat": lat_vals, "lon": lon_vals}, dims=("lat", "lon"))

    area_da = area_da_custom(lat, lon)

    # Masks
    non_urban  = (urban_mask <= 3) | urban_mask.isnull()
    crop_only  = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())

    # biome raster mask on ds grid using
    sample_da = ds[f"{prefix}_ac1_kt"]
    def biome_mask_from_gdf(gdf):
        clipped = clip_da(sample_da, gdf)
        clipped = clipped.interp_like(sample_da, method="nearest")
        return clipped.notnull()

    # Accumulate
    family_acc = {}
    for fam_name, cats in families.items():
        family_acc[fam_name] = {
            g: {"den": 0.0, "cats": {cat_name: 0.0 for cat_name in cats.keys()}}
            for g in groups_plot
        }

    for biome in BIOMES:
        if biome == "Mangroves":  # excluded 
            continue
        grp = GROUP_MAP.get(biome)
        if grp is None:
            continue

        gdf, _ = get_biomes(biome, tnc_shp=str(TNC_SHP))
        bmask = biome_mask_from_gdf(gdf)

        
        denom_mask = land & bmask & (~crop_only)
        denom_area = float(area_da.where(denom_mask).sum().values)
        if denom_area == 0.0:
            continue

        # terrestrial, biome, non-urban, not cropland
        valid_num_mask = land & bmask & non_urban & (~crop_only)

        for fam_name, cats in families.items():
            # add the same denominator for this biome to this family
            family_acc[fam_name][grp]["den"] += denom_area
            # add per-category areas
            for cat_name, cat_mask in cats.items():
                hit_area = float(area_da.where(valid_num_mask & cat_mask).sum().values)
                family_acc[fam_name][grp]["cats"][cat_name] += hit_area

    # Cropland handled separately
    grp = "Cropland"
    crop_denom_area = float(area_da.where(land & rainfed_cropland).sum().values)
    for fam_name, cats in families.items():
        family_acc[fam_name][grp]["den"] += crop_denom_area
        for cat_name, cat_mask in cats.items():
            hit_area = float(area_da.where(land & rainfed_cropland & non_urban & cat_mask).sum().values)
            family_acc[fam_name][grp]["cats"][cat_name] += hit_area

    family_dfs = {}
    for fam_name, cats in families.items():
        rows = []
        cat_names = list(cats.keys())
        for g in groups_plot:
            den = family_acc[fam_name][g]["den"]
            if den <= 0:
                vals = {cn: 0.0 for cn in cat_names}
            else:
                vals = {cn: 100.0 * family_acc[fam_name][g]["cats"][cn] / den for cn in cat_names}
            row = {"Group": g, **vals}
            rows.append(row)
        df = pd.DataFrame(rows)
        df["Group"] = pd.Categorical(df["Group"], categories=groups_plot, ordered=True)
        df = df.sort_values("Group")
        family_dfs[fam_name] = df



    # ----- Plot ------

    sn.set_style("whitegrid")

    # Target size
    MM_TO_INCH = 1.0 / 25.4
    FIG_W_MM = 118
    FIG_H_MM = 35
    FIGSIZE_COMBINED = (FIG_W_MM * MM_TO_INCH, FIG_H_MM * MM_TO_INCH)

    # Create figure 
    fig, axes = plt.subplots(
        1, 3,
        figsize=FIGSIZE_COMBINED,
        sharey=True,
        gridspec_kw=dict(
            left=0.36,   # space for y-tick labels
            right=0.98,
            bottom=0.22, # space for x-tick labels
            top=0.88,    # space for titles
            wspace=0.18  # panels close but not overlapping
        ),
    )

    RED   = "#64875C"
    BLUE  = "#B36E94"
    GREY  = "#574839"


    family_order = ["AC1 & SD", "Fractal dimension", "Flickering (Skew & Kurt)"]
    category_colors = {
        "AC1 & SD": {
            "Both up (AC1 up & SD up)": RED,
            "Mixed (one up, one down)": GREY,
            "Both down (AC1 down & SD down)": BLUE,
        },
        "Fractal dimension": {
            "FD up": RED,
            "FD down": BLUE,
        },
        "Flickering (Skew & Kurt)": {
            "Skew up & Kurt up": RED,
            "Mixed (one up, one down)": GREY,
            "Skew down & Kurt down": BLUE,
        },
    }

    for ax, fam_name in zip(axes, family_order):
        dfp = family_dfs[fam_name].copy()
        cat_names = list(families[fam_name].keys())  # preserve order

        left = np.zeros(len(dfp))
        for cn in cat_names:
            ax.barh(
                y=dfp["Group"],
                width=dfp[cn].to_numpy(),
                left=left,
                color=category_colors[fam_name][cn],
                edgecolor="none",
                height=0.90,  # bars close
            )
            left += dfp[cn].to_numpy()

        ax.set_title(fam_name, fontsize=8, pad=4)

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])

        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=6)

        ax.grid(False)

    # Shared y means inverting on one inverts all
    axes[0].invert_yaxis()

    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    for ax in axes:
        sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
        ax.margins(y=0.02)
      
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(False)

    out_svg = os.path.join(str(outdir), f"{prefix}_kt_biomes_combined_families.svg")
    fig.savefig(
        out_svg,
        format="svg",
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor="white",
    )

    out_pdf = os.path.join(str(outdir), f"{prefix}_kt_biomes_combined_families.pdf")
    fig.savefig(
        out_pdf,
        format="pdf",
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor="white",
    )
    plt.close(fig)

    def clean_label(label: str) -> str:
        return (label.replace("↑", "up")
                    .replace("↓", "down")
                    .replace("↕", "mixed")
                    .replace(" ", "_")) 

    rows_all = []
    for fam_name, dfp in family_dfs.items():
        cat_cols = [c for c in dfp.columns if c != "Group"]
        for _, r in dfp.iterrows():
            for c in cat_cols:
                rows_all.append({
                    "Family": fam_name,
                    "Group": r["Group"],
                    "Category": clean_label(c),
                    "Percent": r[c]
                })
    comb_tidy = pd.DataFrame(rows_all)
    comb_tidy["Group"] = pd.Categorical(comb_tidy["Group"], categories=groups_plot, ordered=True)
    comb_tidy = comb_tidy.sort_values(["Family", "Category", "Group"])

    out_csv_tidy = os.path.join(str(outdir), f"{prefix}_kt_biomes_combined_families_tidy.csv")
    comb_tidy.to_csv(out_csv_tidy, index=False)
    print(f"Saved combined-family percentages (tidy) to: {out_csv_tidy}")


if __name__ == "__main__":
    
    plt.rc("figure", figsize=(13, 8))
    plt.rc("font", size=12)
    sn.set_style("white")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams['svg.fonttype'] = 'none'


    main()
