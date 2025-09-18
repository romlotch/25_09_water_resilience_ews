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
import rioxarray  # for .rio

""" 
Plots bar graphs of the percentage of land area of increasing or decreasing indicators. 
Biomes have been grouped for brevity. 

E.g. 
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr --var 'Et' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr --var 'sm' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
    python 02b-plot_biomes.py --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr --var 'precip' --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2 --mode combined
 
"""

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Grouped biome stacked bars (Decrease/Neutral/Increase) per indicator."
    )
    ap.add_argument("--dataset", required=True, help="Path to NetCDF/Zarr dataset (e.g., out_sm_kt.zarr)")
    ap.add_argument("--var", required=True, help="Variable prefix (e.g., sm, Et, precip)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["indicators", "combined"], default="indicators",
                    help="Plot raw indicators (5 panels) or composite signals (5 panels).")
    return ap.parse_args()

# ---------------- Data & helpers  ----------------
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


from collections import OrderedDict

def combined_family_masks(ds, prefix):
    """
    Families of combined signals with both directions in the same family:
      - AC1 & SD: Both ↑, Mixed (one up one down), Both ↓
      - Fractal dimension: FD ↑, FD ↓
      - Flickering: Skew↑ & Kurt↑, Skew↓ & Kurt↓
    All components must be significant at p < 0.05.
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
    flick_up   = (sig(skew_p) & (skew > 0)) & (sig(kurt_p) & (kurt > 0))
    flick_down = (sig(skew_p) & (skew < 0)) & (sig(kurt_p) & (kurt < 0))

    return OrderedDict([
        ("AC1 & SD", OrderedDict([
            ("Both ↑ (AC1↑ & SD↑)", both_up),
            ("Mixed (one up, one down)", mixed),
            ("Both ↓ (AC1↓ & SD↓)", both_down),
        ])),
        ("Fractal dimension", OrderedDict([
            ("FD ↑", fd_up),
            ("FD ↓", fd_down),
        ])),
        ("Flickering (Skew & Kurt)", OrderedDict([
            ("Skew↑ & Kurt↑", flick_up),
            ("Skew↓ & Kurt↓", flick_down),
        ])),
    ])


def get_biomes(biome):
    fp = TNC_SHP
    data = gpd.read_file(fp, where=f"WWF_MHTNAM='{biome}'")
    unique_ids = [i for i in data.WWF_MHTNAM.unique()]
    unique_num = [i for i in data.WWF_MHTNUM.unique()]
    biomes_df = pd.DataFrame(unique_ids)
    biomes_df['label'] = unique_num
    pd.set_option('display.max_colwidth', None)
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
    """1D lat band area (km²) for each latitude, matching your notebook."""
    R = 6371  # km
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    return (R**2) * dlat_rad * dlon_rad * np.cos(lat_rad)

def area_da_for(clipped_ds: xr.Dataset) -> xr.DataArray:
    """Broadcast that 1D area to 2D (lat, lon), exactly like the notebook."""
    latitudes = clipped_ds['lat'].values
    area_per_lat = get_area_grid(latitudes)
    area_grid = np.broadcast_to(area_per_lat[:, np.newaxis], (latitudes.size, clipped_ds['lon'].size))
    return xr.DataArray(area_grid, coords={'lat': latitudes, 'lon': clipped_ds['lon'].values}, dims=('lat', 'lon'))

def tri_split_areas(kt: xr.DataArray, pval: xr.DataArray, area_da: xr.DataArray):
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

    # dataset + masks (hardcoded paths exactly like your cells)
    ds = xr.open_dataset(args.dataset)
    urban_mask = xr.open_dataset('/mnt/data/romi/data/urban_mask.zarr')
    crop_mask  = xr.open_dataset('/mnt/data/romi/data/crop_mask.zarr')

    urban_mask = urban_mask.rio.write_crs("EPSG:4326")
    crop_mask  = crop_mask.rio.write_crs("EPSG:4326")

    urban_mask = urban_mask.interp_like(ds, method="nearest")
    crop_mask  = crop_mask.interp_like(ds, method="nearest")

    urban_mask = urban_mask['urban-coverfraction'].squeeze('time')
    crop_mask  = crop_mask['crops-coverfraction'].squeeze('time')

    prefix = args.var  # e.g., "sm", "Et", "precip"

    IRRIG_PATH = "/mnt/data/romi/data/driver_analysis/global_irrigated_areas.zarr"
    IRRIG_THRESH_PERCENT = 60.0  # AEI ≥ 60% => considered irrigated (excluded from Cropland pseudo-biome)

    irrig_ds = xr.open_dataset(IRRIG_PATH).rio.write_crs("EPSG:4326").interp_like(ds, method="nearest")
    if len(irrig_ds.data_vars) == 0:
        raise ValueError("Irrigation dataset has no data variables.")
    _irrig_da = list(irrig_ds.data_vars.values())[0].squeeze()
    try:
        _vmax = float(_irrig_da.max().values)
    except Exception:
        _vmax = 1.0
    irrig_frac = _irrig_da / 100.0 if _vmax > 1.5 else _irrig_da

    # base cropland mask (as in your script)
    crop_only = (crop_mask > 25) & ((urban_mask <= 3) | urban_mask.isnull())

    # --- NEW: rain-fed cropland (what we will use for the Cropland pseudo-biome in FAMILIES) ---
    irrigated = (irrig_frac >= IRRIG_THRESH_PERCENT / 100.0)
    rainfed_cropland = crop_only & (~irrigated | irrigated.isnull())
        
    
    
    if args.mode == "indicators":
        # ---------- INDIVIDUAL INDICATORS  ----------
        group_totals = {k: {g: {"dec":0.0,"neu":0.0,"inc":0.0,"tot":0.0} for g in groups_plot} for k,_ in indicators}

        # natural biomes → grouped
        for biome in BIOMES:
            if biome == "Mangroves":
                continue
            grp = GROUP_MAP.get(biome)
            if grp is None:
                continue

            gdf, _ = get_biomes(biome)
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

        # pseudo-biome: cropland
        grp = "Cropland"
        crop_clipped = ds.where(crop_only)
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
        out_svg = os.path.join(args.outdir, f"{prefix}_kt_biomes.svg")
        fig.savefig(out_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)
        


        return


    # ---------- COMBINED SIGNALS (single panels per family; % of total biome land area) ----------
    families = combined_family_masks(ds, prefix)

    # Land-sea mask for denominators
    lsm = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib")
    lsm = lsm.rename({"latitude": "lat", "longitude": "lon"}).rio.write_crs("EPSG:4326")
    lsm = lsm.assign_coords(lon=((lsm.lon + 180) % 360) - 180).sortby("lon")
    lsm = lsm.interp_like(ds, method="nearest")
    land = (lsm["lsm"] > 0.5) | lsm["lsm"].isnull()

    # Area grid (assumes ~regular)
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

    # Helper: biome raster mask on ds grid using any ds var
    sample_da = ds[f"{prefix}_ac1_kt"]
    def biome_mask_from_gdf(gdf):
        clipped = clip_da(sample_da, gdf)
        clipped = clipped.interp_like(sample_da, method="nearest")
        return clipped.notnull()

    # Accumulators: per family → per group → {den, per-category areas}
    family_acc = {}
    for fam_name, cats in families.items():
        family_acc[fam_name] = {
            g: {"den": 0.0, "cats": {cat_name: 0.0 for cat_name in cats.keys()}}
            for g in groups_plot
        }

    # Natural groups (exclude cropland from denom; apply non-urban & non-crop in numerators)
    for biome in BIOMES:
        if biome == "Mangroves":  # excluded like before
            continue
        grp = GROUP_MAP.get(biome)
        if grp is None:
            continue

        gdf, _ = get_biomes(biome)
        bmask = biome_mask_from_gdf(gdf)

        # denominator = land ∧ biome ∧ not cropland (we do not remove urban here, to match your earlier choice)
        denom_mask = land & bmask & (~crop_only)
        denom_area = float(area_da.where(denom_mask).sum().values)
        if denom_area == 0.0:
            continue

        # numerator valid window: terrestrial, biome, non-urban, not cropland
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
    crop_denom_area = float(area_da.where(land & crop_only).sum().values)
    for fam_name, cats in families.items():
        family_acc[fam_name][grp]["den"] += crop_denom_area
        for cat_name, cat_mask in cats.items():
            hit_area = float(area_da.where(land & crop_only & non_urban & cat_mask).sum().values)
            family_acc[fam_name][grp]["cats"][cat_name] += hit_area

    # Build DataFrames: each family becomes a wide DF with columns per category
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

    # ---- Plot: three panels (AC1&SD, FD, Flicker), each stacked like your indicator plots ----
    sn.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Consistent color idea: "↑" = red, "↓" = blue, "Mixed" = grey
    RED   = "#cd5d5dff"
    BLUE  = "#64afc9ff"
    GREY  = "#bab9b9"

    family_order = ["AC1 & SD", "Fractal dimension", "Flickering (Skew & Kurt)"]
    category_colors = {
        "AC1 & SD": {
            "Both ↑ (AC1↑ & SD↑)": RED,
            "Mixed (one up, one down)": GREY,
            "Both ↓ (AC1↓ & SD↓)": BLUE,
        },
        "Fractal dimension": {
            "FD ↑": RED,
            "FD ↓": BLUE,
        },
        "Flickering (Skew & Kurt)": {
            "Skew↑ & Kurt↑": RED,
            "Skew↓ & Kurt↓": BLUE,
        },
    }

    for ax, fam_name in zip(axes, family_order):
        dfp = family_dfs[fam_name].copy()
        cat_names = list(families[fam_name].keys())  # preserves order defined above

        left = np.zeros(len(dfp))
        for cn in cat_names:
            ax.barh(
                y=dfp["Group"],
                width=dfp[cn].to_numpy(),
                left=left,
                color=category_colors[fam_name][cn],
                edgecolor="none",
                height=0.87,
                label=cn
            )
            left += dfp[cn].to_numpy()

        ax.set_title(fam_name, fontsize=13, pad=8)
        ax.set_xlim(0, 100)  
        ax.set_xlabel("% of total biome land area")
        ax.grid(False)

    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="y", labelsize=12)
        sn.despine(ax=ax, top=True, right=True, bottom=True, left=True)
        ax.margins(y=0.02)
        # single legend per panel (optional; comment out if you prefer a single global legend)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    plt.tight_layout()
    out_svg = os.path.join(args.outdir, f"{prefix}_kt_biomes_combined_families.svg")
    fig.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
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

    out_csv_tidy = os.path.join(args.outdir, f"{prefix}_kt_biomes_combined_families_tidy.csv")
    comb_tidy.to_csv(out_csv_tidy, index=False)
    print(f"Saved combined-family percentages (tidy) to: {out_csv_tidy}")



if __name__ == "__main__":
    
    plt.rc("figure", figsize=(13, 8))
    plt.rc("font", size=12)
    sn.set_style("white")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'


    main()
