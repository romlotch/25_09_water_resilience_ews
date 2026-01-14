import os
import argparse
import numpy as np
import xarray as xr
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from scipy import ndimage
from utils.config import load_config, cfg_path
from pathlib import Path

""" 
Plots and saves 5-panel figure of individual indicators and 
three figures with bivariate colourmaps for autocorrelation 
and variance, kurtosis and skewness, and the fractal dimension. 

E.g. 
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr --var sm --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr --var Et --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr --var precip --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/

"""


# ----- figure formatting params for publication ----- 

MM_TO_IN = 1.0 / 25.4

TARGET_W_MM = 80.44     # width in mm
TARGET_H_MM = 39.760    # height in mm

FIGSIZE_SMALL = (
    TARGET_W_MM * MM_TO_IN,  
    TARGET_H_MM * MM_TO_IN   
)

REF_FIG_WIDTH_IN = 20.0


BIV_LEG_W_MM = 14.8    
BIV_LEG_H_MM = 14.6   

FD_LEG_W_MM  = 17.5   
FD_LEG_H_MM  = 5 


def scaled_lw(base_lw: float, ax) -> float:
    """Scale a base linewidth by figure width so small figures get thinner lines."""
    fig_w = ax.figure.get_size_inches()[0]
    scale = fig_w / REF_FIG_WIDTH_IN
    return base_lw * scale



# --- helpers ---

def open_source_ds(path: str) -> xr.Dataset:
    """Open NetCDF or Zarr."""
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zmetadata"))
    ):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def create_bivariate_color(ac1_array, std_array,
                           vmin_ac1=-1, vmax_ac1=1,
                           vmin_std=-1, vmax_std=1,
                           cmap_ac1='RdBu_r', cmap_std='RdBu_r',
                           lighten=0.15):
    """
    Blend two colormaps by RGB averaging,
    then lighten the result toward white.

    lighten: 0.0 keeps original, 0.10–0.25 blends a bit lighter 
    """
    # Normalize to 0..1
    ac1 = np.clip((ac1_array - vmin_ac1) / (vmax_ac1 - vmin_ac1), 0, 1)
    std = np.clip((std_array - vmin_std) / (vmax_std - vmin_std), 0, 1)

    # Sample colormaps 
    c1 = plt.get_cmap(cmap_ac1)(ac1)[..., :3]
    c2 = plt.get_cmap(cmap_std)(std)[..., :3]

    blended_rgb = (c1 + c2) / 2.0

    # Lighten toward white if you want
    if lighten and lighten != 0:
        blended_rgb = np.clip(blended_rgb + lighten * (1.0 - blended_rgb), 0.0, 1.0)

    # Return RGBA 
    blended_rgba = np.concatenate(
        [blended_rgb, np.ones(blended_rgb.shape[:-1] + (1,))], axis=-1
    )
    return blended_rgba


def make_bivariate_lut(
        vmin_a, vmax_a, vmin_b, vmax_b,
        n_bins=8,
        cmap_a='PRGn', cmap_b='PRGn',
        lighten=-0.7
    ):

    """
    Build an n_bins x n_bins RGBA LUT (for A,B) and the bin edges.

    Returns:
        lut : RGBA colors for each (i,j) bin pair.
        a_edges, b_edges : Bin edges used for digitizing A and B.
    """

    # Bin edges
    a_edges = np.linspace(vmin_a, vmax_a, n_bins + 1)
    b_edges = np.linspace(vmin_b, vmax_b, n_bins + 1)

    # Bin centers (what we actually color)
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])

    # Make a grid of centers
    Agrid, Bgrid = np.meshgrid(a_centers, b_centers, indexing='ij')
    lut = create_bivariate_color(
        Agrid, Bgrid,
        vmin_a, vmax_a, vmin_b, vmax_b,
        cmap_ac1=cmap_a, cmap_std=cmap_b,
        lighten=lighten
    )  # shape: (n_bins, n_bins, 4)

    return lut, a_edges, b_edges


def add_base_map(ax):

    lw_coast = scaled_lw(2, ax)  
    ax.add_feature(cfeature.LAND,
                   facecolor='white',
                   linewidth=0.0,
                   edgecolor='none',
                   zorder=0)
    
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=lw_coast,
                   zorder=10)
    
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])


def add_nodata_mask(
        ax, mask_da, lon, lat, extent,
        min_lat=-60.0,
        grey_rgba=(0.94, 0.94, 0.94, 1.0),
        outline_alpha=1,
        z_fill=2, z_tex=2.6, z_outline=3,
        fill_beyond_coverage=True
    ):

    """
    Light-grey fill where mask==True, then overlay one-way black hatch
    using contourf. Thin semi-transparent outline on top.
    """

    grey_da = (mask_da & (mask_da["lat"] > min_lat))
    grey_vals = grey_da.transpose('lat', 'lon').fillna(False).astype(int).values  # 0/1

    # Solid light-grey fill (0=transparent, 1=grey)
    cmap_mask = ListedColormap([(0, 0, 0, 0), grey_rgba])
    ax.imshow(
        grey_vals, origin='upper', extent=extent,
        transform=ccrs.PlateCarree(), interpolation='nearest',
        cmap=cmap_mask, zorder=z_fill
    )
    
    mpl.rcParams.setdefault('hatch.linewidth', 0.25) 

    hatch_coll = ax.contourf(
        lon, lat, grey_vals,            
        levels=[0.5, 1.5],            
        colors='none',                   
        hatches=['////'],                  # one-way hatch
        transform=ccrs.PlateCarree(),
        zorder=z_tex
    )
    # Force hatch color to black and keep fill transparent
    for coll in hatch_coll.collections:
        coll.set_facecolor('none')
        coll.set_edgecolor('black')
        coll.set_linewidth(0.0)         

    # Thin semi-transparent outline around the grey region
    outline = ax.contour(
        lon, lat, grey_vals,
        levels=[0.5],
        colors='black',
        transform=ccrs.PlateCarree(),
        zorder=z_outline
    )
    for lc in outline.collections:
        lc.set_linewidth(scaled_lw(0.5, ax))   
        lc.set_alpha(outline_alpha)    
        lc.set_antialiased(True)

    if not fill_beyond_coverage:
        return
    
    dlat = float(np.median(np.abs(np.diff(lat))))
    lat_max_cov = float(lat.max())

    cap_overlap = 0.01  # degrees
    cap_min_lat = min(90.0, lat_max_cov - cap_overlap)
    
    if cap_min_lat >= 90.0:
        return

    land_path = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_reader = shpreader.Reader(land_path)

    north_cap = box(-180.0, cap_min_lat, 180.0, 90.0)

    for geom in land_reader.geometries():

        inter = geom.intersection(north_cap)

        if inter.is_empty:
            continue

        # Solid fill
        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor=grey_rgba, edgecolor='none', zorder=z_fill
        )

        # Hatch overlay
        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black', hatch='////',
            linewidth=0.25, zorder=z_tex
        )
        # Optional outline 
        ax.add_geometries(
            [inter], crs=ccrs.PlateCarree(),
            facecolor='none', edgecolor='black',
            linewidth=scaled_lw(0.5, ax), alpha=outline_alpha, zorder=z_outline
        )
        

def land_mask_bool_like(ds: xr.Dataset) -> xr.DataArray:
    """Boolean land mask on ds grid using Natural Earth regions."""
    land_regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    lm = land_regions.mask(ds)
    return (~lm.isnull())


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def robust_sym_limits(da: xr.DataArray, q: float = 0.995) -> tuple[float, float]:
    """Symmetric limits around 0 using |da| quantile (robust)."""
    try:
        a = float(np.abs(da).quantile(q, skipna=True))
    except Exception:
        a = float(np.nanquantile(np.abs(da.values), q))
    if not np.isfinite(a) or a == 0.0:
        a = float(np.nanmax(np.abs(da.values))) if np.isfinite(da.values).any() else 1.0
    if not np.isfinite(a) or a == 0.0:
        a = 1.0
    return -a, a


def adjust_cmap_lightness(cmap, lighten=-0.5, N=256):
    """
    Apply the same lightening/darkening transform used in create_bivariate_color
    to a 1D colormap (used in the FD magnitude plot).

    lighten > 0 : moves colors toward white
    lighten < 0 : moves colors toward black
    """
   
    if isinstance(cmap, str):
        base = plt.get_cmap(cmap, N)
    else:
        base = cmap

    # Sample RGBA
    colors = base(np.linspace(0, 1, N))
    rgb = colors[:, :3]

    if lighten and lighten != 0:
        rgb = np.clip(rgb + lighten * (1.0 - rgb), 0.0, 1.0)

    colors[:, :3] = rgb
    return ListedColormap(colors, name=f"{base.name}_lt{lighten}")


# --- Plotting functions ---

def plot_five_panel(ds: xr.Dataset, var_prefix: str, outdir: str,
                    add_mask: bool = True, auto_range: bool = False, q: float = 0.995):
    """
    Make fives separate figures (one per indicator).
    Saves to pic outdir and a standalone SVG colorbar.

    Unchanged. 
    """
    ensure_dir(outdir)
    sn.set_style("white")
    cmap = sn.color_palette("RdBu_r", as_cmap=True)

    suffixes = ["ac1", "std", "skew", "kurt", "fd"]
    labels   = ["AC1", "SD", "Skew.", "Kurt.", "FD"]

    lon = ds['lon'].values
    lat = ds['lat'].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    land_regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land_regions.mask(ds)
    land_mask_bool = ~land_mask.isnull()

    for suf, label in zip(suffixes, labels):
        kt_name = f"{var_prefix}_{suf}_kt"
        pv_name = f"{var_prefix}_{suf}_pval"
        if (kt_name not in ds) or (pv_name not in ds):
            print(f"[warn] missing {kt_name} or {pv_name}; skipping {label}")
            continue

        # p<0.05 and not 0 map
        sig = (ds[pv_name] < 0.05) & (ds[kt_name] != 0)
        da  = ds[kt_name].where(sig)

        # Decide color limits (robust symmetric)
        if auto_range:
            vmin, vmax = robust_sym_limits(da, q=q)
        else:
            vmin, vmax = -1, 1

        # Figure
        fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', linewidth=0.5, zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, color='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        sc = da.plot(
            ax=ax, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, cmap=cmap,
            add_colorbar=False, rasterized=True
        )

        """ if add_mask:
            mask_da = (ds[f"{var_prefix}_ac1_kt"].isnull() & land_mask_bool)
            add_nodata_mask(ax, mask_da, lon, lat, extent, min_lat=-60.0, outline_lw=0.5, outline_alpha=1, zorder=1.5)
        """
        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_title("")  # keep clean
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.xaxis.set_ticks([]); ax.yaxis.set_ticks([])
        

        # Save map
        base = f"{var_prefix}_kt_{label.lower()}"
        outfile = os.path.join(outdir, f"{base}.png")
        fig.savefig(outfile, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Standalone colorbar (SVG)
        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb, orientation='horizontal'
        )
        cb1.set_label(f"{label} Kendall tau")
        cb1.ax.tick_params(labelsize=8)
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)
        colorbar_outfile = os.path.join(outdir, f"colorbar_{base}.svg")
        fig_cb.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f"Saved: {outfile}")


def plot_bivariate(ds: xr.Dataset, var_prefix: str, a: str, b: str, outdir: str,
                   vmin_a=-1, vmax_a=1, vmin_b=-1, vmax_b=1,
                   cmap_a='PRGn', cmap_b='PRGn',
                   add_mask=True):
    
    """Bivariate map for (a,b) with legend."""

    # significance filters
    sig_a = ds[f"{var_prefix}_{a}_kt"].where(ds[f"{var_prefix}_{a}_pval"] < 0.05)
    sig_b = ds[f"{var_prefix}_{b}_kt"].where(ds[f"{var_prefix}_{b}_pval"] < 0.05)
    both = (~sig_a.isnull()) & (~sig_b.isnull())
    A = sig_a.where(both).values  
    B = sig_b.where(both).values

    lon = ds['lon'].values
    lat = ds['lat'].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    land_regions = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land_regions.mask(ds)
    land_mask_bool = ~land_mask.isnull()

    #  build 8×8 LUT and discretize A,B
    n_bins = 8
    lut, a_edges, b_edges = make_bivariate_lut(
        vmin_a, vmax_a, vmin_b, vmax_b,
        n_bins=n_bins,
        cmap_a=cmap_a, cmap_b=cmap_b,
        lighten=-0.5
    )

    # Digitize values into bin indices
    a_idx = np.digitize(A, a_edges) - 1   # 0..n_bins
    b_idx = np.digitize(B, b_edges) - 1

    # Make sure indices are in range
    a_idx = np.clip(a_idx, 0, n_bins - 1)
    b_idx = np.clip(b_idx, 0, n_bins - 1)

    # Build RGBA color array for the map (same shape as A,B)
    colors = np.zeros(A.shape + (4,), dtype=float)

    # Valid where we have finite values and land
    alpha_mask = np.isfinite(A) & np.isfinite(B) & land_mask_bool.values


    alpha_raw = np.isfinite(A) & np.isfinite(B) & land_mask_bool.values
    alpha_filled = ndimage.binary_fill_holes(alpha_raw)
    hole_mask = alpha_filled & (~alpha_raw) & land_mask_bool.values

    structure = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]])  

    labeled, ncomp = ndimage.label(alpha_mask, structure=structure)
    sizes = ndimage.sum(alpha_mask, labeled, index=range(1, ncomp + 1))
    min_size = 5  
    keep_labels = np.where(sizes >= min_size)[0] + 1  # labels start at 1

    keep_mask = np.isin(labeled, keep_labels)


    alpha_mask_filtered = alpha_mask & keep_mask
    colors[alpha_mask_filtered] = lut[a_idx[alpha_mask_filtered], b_idx[alpha_mask_filtered]]

    # Set alpha 1 for valid, 0 otherwise
    colors[alpha_mask_filtered, 3] = 1.0
    colors[~alpha_mask_filtered, 3] = 0.0

    colors = np.zeros(A.shape + (4,), dtype=float)
    colors[..., 3] = 0.0

    # Significant pixels full color, full alpha
    colors[alpha_raw] = lut[a_idx[alpha_raw], b_idx[alpha_raw]]
    colors[alpha_raw, 3] = 1.0

    # “Hole” pixels (no data) neutral center color, low alpha
    n_bins = lut.shape[0]
    neutral_color = lut[n_bins // 2, n_bins // 2].copy()  # center of the bivariate palette

    colors[hole_mask, :3] = neutral_color[:3]
    colors[hole_mask, 3]  = 0.3 


    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.imshow(colors, origin='upper', extent=extent,
              transform=ccrs.PlateCarree(), interpolation='nearest', zorder=1)
    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())

    if add_mask:
        lw_outline = scaled_lw(0.5, ax)
        mask_da = (ds[f"{var_prefix}_ac1_kt"].isnull() & land_mask_bool)
        add_nodata_mask(
            ax, mask_da, lon, lat, extent,
            min_lat=-60.0, outline_alpha=1,
            fill_beyond_coverage=True
        )

    plt.tight_layout()
    ensure_dir(outdir)
    fname = os.path.join(outdir, f"{var_prefix}_bivar_{a}_{b}.png")
    fig.savefig(fname, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Legend with discrete bins

    n_bins = 7  # 7 × 7 legend

    # Bin edges and centers for each axis
    a_edges = np.linspace(vmin_a, vmax_a, n_bins + 1)
    b_edges = np.linspace(vmin_b, vmax_b, n_bins + 1)

    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    b_centers = 0.5 * (b_edges[:-1] + b_edges[1:])

    # 7×7 grid of bin centers
    Agrid, Bgrid = np.meshgrid(a_centers, b_centers, indexing='ij')

    legend_colors = create_bivariate_color(
        Agrid, Bgrid,
        vmin_a, vmax_a, vmin_b, vmax_b,
        cmap_a, cmap_b,
        lighten=-0.5
    )

    # Fixed-size legend figure: 14.8 mm × 14.6 mm (including text)
    fig2 = plt.figure(
        figsize=(
            BIV_LEG_W_MM * MM_TO_IN,
            BIV_LEG_H_MM * MM_TO_IN
        )
    )
    # Leaving room for tick labels
    ax2 = fig2.add_axes([0.35, 0.30, 0.6, 0.6])  # [left, bottom, width, height] in figure fraction

    ax2.imshow(
        legend_colors,
        origin='lower',
        extent=[vmin_b, vmax_b, vmin_a, vmax_a],
        interpolation='nearest'
    )
    ax2.set_xlabel(f"{b.upper()} trend", fontsize=6)
    ax2.set_ylabel(f"{a.upper()} trend", fontsize=6)
    ax2.set_xticks([vmin_b, 0, vmax_b])
    ax2.set_yticks([vmin_a, 0, vmax_a])
    ax2.tick_params(axis="both", labelsize=5)
    ax2.set_title("")
    for spine in ax2.spines.values():
        spine.set_visible(True)

    leg_fname = os.path.join(outdir, f"{var_prefix}_bivar_{a}_{b}_legend.svg")
    fig2.savefig(
        leg_fname,
        format="svg",
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor='white'
    )
    plt.close(fig2)

def plot_fd_magnitude(ds: xr.Dataset, var_prefix: str, outdir: str, cmap, lighten = None):

    """FD magnitude map: positive = increase, negative = decrease, p<0.05."""

    sn.set_style("white")
    kt = ds[f"{var_prefix}_fd_kt"]
    pv = ds[f"{var_prefix}_fd_pval"]

    sig = pv < 0.05
    inc = (kt > 0) & sig
    dec = (kt < 0) & sig
    change_mask = inc | dec

    mag = (abs(kt).where(inc, 0) - abs(kt).where(dec, 0)).where(change_mask, np.nan)

    land_mask_bool = land_mask_bool_like(ds)
    mag = mag.where(land_mask_bool)
    
    norm = mpl_colors.TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

    lon = ds['lon'].values
    lat = ds['lat'].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    if lighten is not None:
        cmap_obj = adjust_cmap_lightness(cmap, lighten=lighten)
    else:
        # original behaviour
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')

    # plot data
    qm = mag.plot(
        ax=ax, cmap=cmap_obj, norm=norm, transform=ccrs.PlateCarree(),
        add_colorbar=False, add_labels=False, rasterized=True
    )

    ax.set_extent([-145, 180, -60, 90], crs=ccrs.PlateCarree())

    # overlay mask using AC1 NaNs 
    lw_outline = scaled_lw(2, ax)
    mask_da = (ds[f"{var_prefix}_ac1_kt"].isnull() & land_mask_bool)
    add_nodata_mask(
            ax, mask_da, lon, lat, extent,
            min_lat=-60.0, outline_alpha=1,
            fill_beyond_coverage=True
        )
    
    plt.tight_layout()
    ensure_dir(outdir)
    map_outfile = os.path.join(outdir, f"{var_prefix}_fd_magnitude.png")
    fig.savefig(map_outfile, dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # colorbar
    fig_cb = plt.figure(
        figsize=(
            FD_LEG_W_MM * MM_TO_IN,
            FD_LEG_H_MM * MM_TO_IN
        )
    )
    # Make a narrow horizontal axis across most of the figure
    ax_cb = fig_cb.add_axes([0.15, 0.4, 0.7, 0.4])  # [left, bottom, width, height]

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cb = plt.colorbar(
        sm,
        cax=ax_cb,
        orientation='horizontal',
    )

    cb.set_label('FD trend magnitude (+ increase, − decrease)', fontsize=5)
    cb.ax.tick_params(labelsize=4, length=2)
    cb.outline.set_linewidth(0.3333)

    cbar_outfile = os.path.join(outdir, f"colorbar_{var_prefix}_fd_magnitude.svg")
    fig_cb.savefig(
        cbar_outfile,
        format='svg',
        dpi=450,
        bbox_inches=None,
        pad_inches=0.0,
        transparent=True
    )
    plt.close(fig_cb)


# --- Main ---

def main():
    p = argparse.ArgumentParser(description="Make maps of Kendall Tau output.")
    p.add_argument("--var", required=True, help="Variable prefix (e.g., sm, Et, precip)")
    p.add_argument("--suffix", default=None,
                help="Optional suffix (e.g. breakpoint_stc). Used to infer dataset path if --dataset not provided.")
    p.add_argument("--dataset", default=None,
                help="Optional override path to *_kt.zarr. If omitted, inferred from config + --var + --suffix.")
    p.add_argument("--outdir", default=None,
                help="Optional override output directory. If omitted, uses outputs/figures/kt/<var>/ under outputs_root.")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)


    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)

    def _sfx(s):
        if not s: return ""
        return s if str(s).startswith("_") else f"_{s}"

    # default dataset: outputs/zarr/out_<var><_suffix>_kt.zarr
    default_ds = Path(outputs_root) / "zarr" / f"out_{args.var}{_sfx(args.suffix)}_kt.zarr"
    ds_path = Path(args.dataset) if args.dataset else default_ds

    # default outdir: outputs/figures/kt/<var>/
    default_outdir = Path(outputs_root) / "figures" / "kt" / args.var
    outdir = Path(args.outdir) if args.outdir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ds = open_source_ds(str(ds_path))


    """ if args.var == "precip":
        mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
        ds_mask = xr.open_dataset(mask_path, engine="cfgrib")

        ds_mask = ds_mask.rio.write_crs('EPSG:4326')
        ds_mask = ds_mask.assign_coords(longitude=(((ds_mask.longitude + 180) % 360) - 180)).sortby('longitude')
        ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)
        
        ds_mask = ds_mask.interp(
            longitude=ds.lon,   
            latitude=ds.lat,
            method="nearest"
        )
        
        mask = ds_mask["lsm"] > 0.7
        ds = ds.where(mask) """

    # 1) individual indicators
    plot_five_panel(ds, args.var, str(outdir))

    # 2) bivariate AC1 & STD (+ legend)
    plot_bivariate(ds, args.var, 'ac1', 'std', str(outdir), cmap_a=cmap, cmap_b=cmap)

    # 3) bivariate Skew & Kurt (+ legend)
    plot_bivariate(ds, args.var, 'skew', 'kurt', str(outdir), cmap_a=cmap, cmap_b=cmap)

    # 4) FD map
    plot_fd_magnitude(ds, args.var, str(outdir),  cmap = cmap, lighten = -0.5)


if __name__ == "__main__":

    pink  = "#82315E"  
    green = "#256D15" 

    # Slightly desaturate 
    pink_soft  = sn.desaturate(pink, 0.95)
    green_soft = sn.desaturate(green, 0.95)

    cmap = sn.blend_palette(
        [pink_soft, "#f5f5f5", green_soft], as_cmap=True
    )
    
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['hatch.linewidth'] = 0.08   
    mpl.rcParams['lines.solid_joinstyle'] = 'round'
    mpl.rcParams['lines.solid_capstyle']  = 'round'

    main()