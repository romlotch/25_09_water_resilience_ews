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

""" 
Plots and saves 5-panel figure of individual indicators and 
three figures with bivariate colourmaps for autocorrelation 
and variance, kurtosis and skewness, and the fractal dimension. 

E.g. 
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_sm_final/out_sm_kt.zarr --var sm --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_Et_final/out_Et_kt.zarr --var Et --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/
    python 02a-plot_kt.py --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip_kt.zarr --var precip --outdir /mnt/data/romi/figures/paper_1/results_final/figure_2/

"""

# ==== helpers ====

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
    Blend two colormaps by simple RGB averaging (same look as before),
    then optionally lighten the result toward white.

    lighten: 0.0 keeps original; 0.10–0.25 blends a bit lighter without
             changing the hue.
    """
    # Normalize to 0..1
    ac1 = np.clip((ac1_array - vmin_ac1) / (vmax_ac1 - vmin_ac1), 0, 1)
    std = np.clip((std_array - vmin_std) / (vmax_std - vmin_std), 0, 1)

    # Sample colormaps (sRGB)
    c1 = plt.get_cmap(cmap_ac1)(ac1)[..., :3]
    c2 = plt.get_cmap(cmap_std)(std)[..., :3]

    # Original look: simple average blend
    blended_rgb = (c1 + c2) / 2.0

    # Lighten toward white (optional)
    if lighten and lighten != 0:
        blended_rgb = np.clip(blended_rgb + lighten * (1.0 - blended_rgb), 0.0, 1.0)

    # Return RGBA (alpha handled separately by your mask)
    blended_rgba = np.concatenate(
        [blended_rgb, np.ones(blended_rgb.shape[:-1] + (1,))], axis=-1
    )
    return blended_rgba


def add_base_map(ax):
    ax.add_feature(cfeature.LAND, facecolor='white', linewidth=0.5, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)
    # Clean frame & ticks
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def add_nodata_mask(
    ax, mask_da, lon, lat, extent,
    min_lat=-60.0,
    grey_rgba=(0.94, 0.94, 0.94, 1.0),
    outline_lw=0.5, outline_alpha=1,
    z_fill=2, z_tex=2.6, z_outline=3,
    fill_beyond_coverage=True
):
    """
    Light-grey fill where mask==True, then overlay a classic one-way black hatch
    using contourf (no RGBA texture). Thin semi-transparent outline on top.
    """
    # Restrict by latitude and ensure boolean mask
    grey_da = (mask_da & (mask_da["lat"] > min_lat))
    grey_vals = grey_da.transpose('lat', 'lon').fillna(False).astype(int).values  # 0/1

    dlat = float(np.median(np.abs(np.diff(lat)))) if len(lat) > 1 else 0.25
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min_cov, lat_max_cov = float(lat.min()), float(lat.max())


    # 1) Solid light-grey fill (0=transparent, 1=grey)
    cmap_mask = ListedColormap([(0, 0, 0, 0), grey_rgba])
    extent_overlap = [lon_min, lon_max, lat_min_cov, lat_max_cov + 0.5 * dlat]
    ax.imshow(
        grey_vals, origin='upper', extent=extent_overlap,
        transform=ccrs.PlateCarree(), interpolation='nearest',
        cmap=cmap_mask, zorder=z_fill
    )
    
    mpl.rcParams.setdefault('hatch.linewidth', 0.25) 

    hatch_coll = ax.contourf(
        lon, lat, grey_vals,             # lon/lat are 1D; Z is 2D (lat,lon)
        levels=[0.5, 1.5],               # selects the "1" region
        colors='none',                   # no solid fill here
        hatches=['//'],                  # one-way hatch
        transform=ccrs.PlateCarree(),
        zorder=z_tex
    )
    # Force hatch color to black and keep fill transparent
    for coll in hatch_coll.collections:
        coll.set_facecolor('none')
        coll.set_edgecolor('black')
        coll.set_linewidth(0.0)         

    # 3) Thin semi-transparent outline around the grey region
    outline = ax.contour(
        lon, lat, grey_vals,
        levels=[0.5],
        colors='black',
        transform=ccrs.PlateCarree(),
        zorder=z_outline
    )
    for lc in outline.collections:
        lc.set_linewidth(outline_lw)   
        lc.set_alpha(outline_alpha)    
        lc.set_antialiased(True)

    if not fill_beyond_coverage:
        return
    
    dlat = float(np.median(np.abs(np.diff(lat))))
    lat_max_cov = float(lat.max())

    cap_overlap = max(0.6 * dlat, 0.15) 
    cap_min_lat = min(90.0, lat_max_cov - cap_overlap)
    if cap_min_lat >= 90.0:
        return

    grey_vals_outline = grey_vals.copy()
    if grey_vals_outline.shape[0] >= 1:
        grey_vals_outline[-1, :] = 0  # drop top edge only

    """ outline = ax.contour(
        lon, lat, grey_vals_outline,
        levels=[0.5],
        colors='black',
        transform=ccrs.PlateCarree(),
        zorder=z_outline
    )
    for lc in outline.collections:
        lc.set_linewidth(outline_lw)
        lc.set_alpha(outline_alpha)
        lc.set_antialiased(False) """

    land_path = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    land_reader = shpreader.Reader(land_path)

    north_cap = box(-180.0, cap_min_lat, 180.0, 90.0)

    for geom in land_reader.geometries():
        inter = geom.intersection(north_cap)
        if inter.is_empty:
            continue
   
        ax.add_geometries([inter], crs=ccrs.PlateCarree(),
                          facecolor=grey_rgba, edgecolor='none', zorder=z_fill)
        ax.add_geometries([inter], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', hatch='//',
                          linewidth=0, zorder=z_tex)
        

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


# ==== Plotting functions ====

def plot_five_panel(ds: xr.Dataset, var_prefix: str, outdir: str,
                    add_mask: bool = True, auto_range: bool = False, q: float = 0.995):
    """
    make fives separate figures (one per indicator).
    Saves to pic outdir and a standalone SVG colorbar.
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

        # Colorbar (per-figure)
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
        cbar.set_label(f"{label} Kendall τ")

        # Standalone colorbar (SVG)
        fig_cb, ax_cb = plt.subplots(figsize=(4, 0.4))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb, orientation='horizontal'
        )
        cb1.set_label(f"{label} Kendall τ")
        cb1.ax.tick_params(labelsize=8)
        fig_cb.subplots_adjust(bottom=0.5, top=0.9, left=0.05, right=0.95)
        colorbar_outfile = os.path.join(outdir, f"colorbar_{base}.svg")
        fig_cb.savefig(colorbar_outfile, format='svg', dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig_cb)

        print(f"Saved: {outfile}")


def plot_bivariate(ds: xr.Dataset, var_prefix: str, a: str, b: str, outdir: str,
                   vmin_a=-1, vmax_a=1, vmin_b=-1, vmax_b=1,
                   cmap_a='RdBu_r', cmap_b='RdBu_r',
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

    colors = create_bivariate_color(
        A, B, vmin_a, vmax_a, vmin_b, vmax_b, cmap_a, cmap_b
    )
    # alpha only where both significant AND on land
    alpha_mask = np.isfinite(A) & np.isfinite(B) & land_mask_bool.values
    colors[..., 3] = alpha_mask.astype(float)

    fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    ax.imshow(colors, origin='upper', extent=extent,
              transform=ccrs.PlateCarree(), interpolation='none', zorder=1)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    if add_mask:
        # mask where AC1 is NaN (common in your workflow)
        mask_da = (ds[f"{var_prefix}_ac1_kt"].isnull() & land_mask_bool)
        add_nodata_mask(
            ax, mask_da, lon, lat, extent,
            min_lat=-60.0, outline_lw=0.5, outline_alpha=1,
            fill_beyond_coverage=True
        )

    plt.tight_layout()
    ensure_dir(outdir)
    fname = os.path.join(outdir, f"{var_prefix}_bivar_{a}_{b}.png")
    fig.savefig(fname, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Legend
    grid = 100
    avals = np.linspace(vmin_a, vmax_a, grid)
    bvals = np.linspace(vmin_b, vmax_b, grid)
    Agrid, Bgrid = np.meshgrid(avals, bvals, indexing='ij')
    legend_colors = create_bivariate_color(Agrid, Bgrid, vmin_a, vmax_a, vmin_b, vmax_b, cmap_a, cmap_b)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(legend_colors, origin='lower', extent=[vmin_b, vmax_b, vmin_a, vmax_a])
    ax2.set_xlabel(f"{b.upper()} trend", fontsize=12)
    ax2.set_ylabel(f"{a.upper()} trend", fontsize=12)
    ax2.set_xticks([vmin_b, 0, vmax_b]); ax2.set_yticks([vmin_a, 0, vmax_a])
    ax2.set_title("")
    for spine in ax2.spines.values():
        spine.set_visible(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, f"{var_prefix}_bivar_{a}_{b}_legend.png"),
                 dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig2)


def plot_fd_magnitude(ds: xr.Dataset, var_prefix: str, outdir: str):
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

    vmax = float(np.nanmax(np.abs(mag.values))) if np.isfinite(mag.values).any() else 1.0
    norm = mpl_colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r').copy()

    lon = ds['lon'].values
    lat = ds['lat'].values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.Robinson()})
    add_base_map(ax)
    ax.set_global()
    fig.patch.set_facecolor('white')
    fig.set_facecolor('white')

    # plot data
    qm = mag.plot(
        ax=ax, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
        add_colorbar=False, add_labels=False, rasterized=True
    )

    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # overlay mask using AC1 NaNs (as per your notebook)
    mask_da = (ds[f"{var_prefix}_ac1_kt"].isnull() & land_mask_bool)
    add_nodata_mask(
            ax, mask_da, lon, lat, extent,
            min_lat=-60.0, outline_lw=0.5, outline_alpha=1,
            fill_beyond_coverage=True
        )

    # colorbar
    cbar = fig.colorbar(qm, ax=ax, orientation='horizontal', shrink=0.6)
    cbar.set_label('FD trend magnitude (+ increase, − decrease)')

    plt.tight_layout()
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, f"{var_prefix}_fd_magnitude.png"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)



# ==== Main =====

def main():
    parser = argparse.ArgumentParser(description="Make maps of Kendall Tau output.")
    parser.add_argument("--dataset", required=True, help="Path to NetCDF/Zarr dataset (e.g., out_sm_kt.zarr)")
    parser.add_argument("--var", required=True, help="Variable prefix (e.g., sm, Et, precip)")
    parser.add_argument("--outdir", required=True, help="Output directory for figures")
    args = parser.parse_args()

    ds = open_source_ds(args.dataset)

    """ if args.var == "precip":
        ds_mask = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")

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
    plot_five_panel(ds, args.var, args.outdir)

    # 2) bivariate AC1 & STD (+ legend)
    plot_bivariate(ds, args.var, 'ac1', 'std', args.outdir)

    # 3) bivariate Skew & Kurt (+ legend)
    plot_bivariate(ds, args.var, 'skew', 'kurt', args.outdir)

    # 4) FD map
    plot_fd_magnitude(ds, args.var, args.outdir)


if __name__ == "__main__":
    
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['hatch.linewidth'] = 0.08   
    mpl.rcParams['lines.solid_joinstyle'] = 'round'
    mpl.rcParams['lines.solid_capstyle']  = 'round'

    main()