
import os
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

import dask
import rasterio 
import rioxarray  


"""
Driver Data Pipeline 

Formats and consolidates driving environmental variables 
for the explanatory ML model. Some come from ERA5, others from GEE. 
  - Opens monthly ERA5 GRIB datasets (temperature, precipitation, soil moisture,
    evaporation from bare soil / "transpiration", wind u/v, etc.)
    - Computes mean, std, linear monthly trend 
    - Saves out to Zarr 
  - Processes groundwater table depth datasets and harmonizes to a global grid
  - Computes ENSO–precip correlation
  - Converts FAO GMIA ASC irrigated area to Zarr and interpolates to 0.25°
  - Restructures Land Cover features GeoTIFF (category + change) and saves to Zarr
  - Uses GEE to process and export additional drivers (NDVI, GPP, PDSI, Tree/Non-tree cover, Soil texture, Elevation, Fire, Croplands).
    But these require authorisation so are disabled by default. If enabled they'll be saved to the usesr own google drive. 

All filepaths are hardcoded (in caps). Datasets where values contained NaN in places where there isn't data,
NaN's were set to 0 where appropriate (e.g. non-tree cover in forests and vice versa) so those pixels aren't 
filtered out when prepping data for the ML model. 

"""


# Configuration flags 


RUN_ERA5_T2M = True
RUN_ERA5_PRECIP = True
RUN_ERA5_SOIL_MOISTURE = True
RUN_ERA5_TRANSPIRATION = True
RUN_PET = True
RUN_ARIDITY = True
RUN_WIND = True
RUN_GROUNDWATER = True
RUN_ENSO_CORRELATION = True
RUN_IRRIGATED_AREA = True
RUN_LULCC_RESTRUCTURE = True

# Google Earth Engine exports (requires authorisation)
RUN_GEE_PDSI = False
RUN_GEE_GPP = False
RUN_GEE_TREE_COVER = False
RUN_GEE_NON_TREE_COVER = False
RUN_GEE_SOIL_TEXTURE = False
RUN_GEE_ELEVATION = False
RUN_GEE_CROPLANDS = False
RUN_GEE_NDVI = False
RUN_GEE_FIRE = False

# -----------------------------------------------------------------------------
# Files paths
# -----------------------------------------------------------------------------

# ERA5 monthly GRIB folders
ERA5_ROOT = Path("/mnt/data/romi/data/ERA5_0.25_monthly")
PATH_T2M = ERA5_ROOT / "2m_temperature"
PATH_TP = ERA5_ROOT / "total_precipitation"
PATH_SWVL1 = ERA5_ROOT / "volumetric_soil_water_layer_1"
PATH_EVABS = ERA5_ROOT / "evaporation_from_bare_soil"
PATH_U10 = ERA5_ROOT / "10m_u_component_of_wind"
PATH_V10 = ERA5_ROOT / "10m_v_component_of_wind"
PATH_BLH = ERA5_ROOT / "boundary_layer_height"
PATH_CAPE = ERA5_ROOT / "convective_available_potential_energy"
PATH_ALBEDO = ERA5_ROOT / "forecast_albedo"

# Output directories
OUT_DIR_FINAL = Path("/mnt/data/romi/data/driver_analysis_final")
OUT_DIR = Path("/mnt/data/romi/data/driver_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)

# PET netcdf (daily)
PET_DIR = Path("/mnt/data/romi/data/et_pot")
PET_FILES = sorted(glob(str(PET_DIR / "*_daily_pet.nc")))

# Groundwater inputs
GW_FILES = [
    "/mnt/data/romi/data/GW_table_depth/AFRICA_WTD_monthlymeans.nc",
    "/mnt/data/romi/data/GW_table_depth/EURASIA_WTD_monthlymeans.nc",
    "/mnt/data/romi/data/GW_table_depth/NAMERICA_WTD_monthlymeans.nc",
    "/mnt/data/romi/data/GW_table_depth/OCEANIA_WTD_monthlymeans.nc",
    "/mnt/data/romi/data/GW_table_depth/SAMERICA_WTD_monthlymeans.nc",
]
GW_OUT = "/mnt/data/romi/data/driver_analysis/driver_groundwater_table.zarr"

# ENSO
ENSO_CSV = "/mnt/data/romi/data/enso_index.csv"

# FAO irrigated areas
GMIA_ASC = "/mnt/data/romi/data/gmia_v5_aai_pct_aei.asc"
GMIA_ZARR = "/mnt/data/romi/data/driver_analysis/global_irrigated_areas.zarr"

# GEE exports outfile
GEE_EXPORT_DIR = Path("/mnt/data/romi/data/GEE_exports")
GEE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Land cover features (GeoTIFF from GEE export)
LULCC_TIF = str(GEE_EXPORT_DIR / "Land_Cover_Features.tif")
LULCC_ZARR = "/mnt/data/romi/data/driver_analysis/lulcc.zarr"

# Time slice
TIME_START = "2000-01-01"
TIME_END = "2023-12-31"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def open_grib_files(folder):
   
    files = sorted(glob(str(folder / "*.grib")))
    if not files:
        raise FileNotFoundError(f"No .grib files found under: {folder}")
    datasets = []
    for f in files:
        ds = xr.open_dataset(f, engine="cfgrib")
        datasets.append(ds)
    return xr.concat(datasets, dim="time")


def wrap_sort_lon(ds, lon_name):
    
    if lon_name in ds.coords:
        ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180})
        ds = ds.sortby(lon_name)
    return ds


def ensure_lat_lon(ds):

    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    return ds


def to_numeric_months(time_coord):
    """Convert time coordinate to months since start for linear trend."""
    time_in_days = (time_coord - time_coord.isel(time=0)) / np.timedelta64(1, "D")
    return time_in_days / 30.44


def compute_stats_and_trend(ds, var):
    """Compute mean, std, and monthly trend for ds[var]."""
    if var not in ds:
        raise KeyError(f"Variable '{var}' not in dataset. Available: {list(ds.data_vars)}")

    da = ds[var]

    # Use numeric months for linear trend
    months = to_numeric_months(ds.time)
    da_numt = da.assign_coords(time=months)

    coeffs = da_numt.polyfit(dim="time", deg=1)
    slope = coeffs.polyfit_coefficients.sel(degree=1)

    mean_ = da.mean(dim="time")
    std_ = da.std(dim="time")

    out = xr.Dataset(
        {"mean": mean_, "std": std_, "monthly_trend": slope}
    )
    return out


def finalize_geospatial(ds):
 
    ds = ensure_lat_lon(ds)
    ds = wrap_sort_lon(ds, "lon")
    ds = ds.rio.write_crs("EPSG:4326")
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    return ds


def process_and_datasets(ds, variable, output_zarr):
    """Select time, compute stats/trend, and save to Zarr."""
    ds = ensure_lat_lon(ds)
    ds = ds.sel(time=slice(TIME_START, TIME_END))

    out = compute_stats_and_trend(ds, variable)

    out = out.chunk({"lat": 100, "lon": 100})

    out = finalize_geospatial(out)

    print(f"Saving to Zarr: {output_zarr}")
    out.to_zarr(output_zarr, mode="w", compute=True)

    return out


def standardize_lat_lon(ds):
    """Standardize dimension names to lat/lon if needed."""
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds


def process_and_coarsen_pet(filepath, variable, target_res = 0.25):
    """
    Pre-processes PET ds: coarsens to 0.25, resamples daily to monthly mean,
    returns monthly. 
    """
    print(f"Processing PET: {filepath}")
    ds = xr.open_dataset(filepath, chunks={"time": 100})
    ds = standardize_lat_lon(ds)

    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in PET file: {list(ds.data_vars)}")

    # Mask invalid values if present
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ds = ds.where(ds[variable] < 9999)

    # Coarsen to target res
    lat_step = float(abs(ds.lat[1] - ds.lat[0]))
    lon_step = float(abs(ds.lon[1] - ds.lon[0]))
    lat_factor = int(target_res / lat_step) if lat_step < target_res else 1
    lon_factor = int(target_res / lon_step) if lon_step < target_res else 1

    if lat_factor > 1 or lon_factor > 1:
        coarsen_kwargs = {}
        if lat_factor > 1:
            coarsen_kwargs["lat"] = lat_factor
        if lon_factor > 1:
            coarsen_kwargs["lon"] = lon_factor
        ds = ds.coarsen(**coarsen_kwargs, boundary="trim").mean()

    # Resample to monthly 
    ds = ds.resample(time="1MS").mean()
    return ds


def process_concatenate_pet(filepaths, variable, output_zarr):
  
    datasets = [process_and_coarsen_pet(fp, variable) for fp in filepaths]
    print("Concatenating PET datasets")
    combined = xr.concat(datasets, dim="time")
    combined = combined.chunk({"time": 12, "lat": 100, "lon": 100})

    # Compute stats + trend
    out = compute_stats_and_trend(combined, variable)
    out = out.chunk({"lat": 100, "lon": 100})
    out = finalize_geospatial(out)

    print(f"Saving PET stats to Zarr: {output_zarr}")
    out.to_zarr(output_zarr, mode="w", compute=True)
    return out


def process_groundwater_file(filepath, variable = "WTD", target_res = 0.25): 
    """Process one groundwater table depth file: coarsen and compute stats + trend."""
    print(f"Processing Groundwater: {filepath}")
    ds = xr.open_dataset(filepath, chunks={"lat": 100, "lon": 100})
    ds = ensure_lat_lon(ds)

    # Coarsen 
    lat_step = float(abs(ds.lat[1] - ds.lat[0]))
    lon_step = float(abs(ds.lon[1] - ds.lon[0]))
    lat_factor = int(target_res / lat_step) if lat_step < target_res else 1
    lon_factor = int(target_res / lon_step) if lon_step < target_res else 1
    if lat_factor > 1 or lon_factor > 1:
        coarsen_kwargs = {}
        if lat_factor > 1:
            coarsen_kwargs["lat"] = lat_factor
        if lon_factor > 1:
            coarsen_kwargs["lon"] = lon_factor
        ds = ds.coarsen(**coarsen_kwargs, boundary="trim").mean()

    # Trend uses numeric months
    months = to_numeric_months(ds.time)
    da = ds[variable].assign_coords(time=months)
    coeffs = da.polyfit(dim="time", deg=1, cov=False)
    slope = coeffs.polyfit_coefficients.sel(degree=1)

    mean_ = ds[variable].mean(dim="time")
    std_ = ds[variable].std(dim="time")

    out = xr.Dataset({"mean": mean_, "std": std_, "monthly_trend": slope})
  
    out = out.chunk({"lat": -1, "lon": -1})
    return out


def harmonize_grid(datasets):
    """Interpolate all datasets to 0.25 """
    all_lats = np.concatenate([ds.lat.values for ds in datasets])
    all_lons = np.concatenate([ds.lon.values for ds in datasets])

    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()

    common_lats = np.arange(np.floor(lat_min * 4) / 4, np.ceil(lat_max * 4) / 4 + 0.25, 0.25)
    common_lons = np.arange(np.floor(lon_min * 4) / 4, np.ceil(lon_max * 4) / 4 + 0.25, 0.25)

    regridded = [ds.interp(lat=common_lats, lon=common_lons) for ds in datasets]
    regridded = [ds.assign_coords(lat=common_lats, lon=common_lons) for ds in regridded]
    return regridded


def enso_precip_correlation(enso_csv, precip_ds, precip_var = "tp"):
    """Compute correlation between monthly ENSO index and ERA5 precipitation."""
    print("Computing ENSO–precip correlation")
    enso_df = pd.read_csv(enso_csv, delimiter=";", header=None)
    years = enso_df.iloc[:, 0].values
    enso_vals = enso_df.iloc[:, 1:].values.flatten()

    # build monthly datetime index
    months = np.tile(np.arange(1, 13), len(years))
    years_rep = np.repeat(years, 12)
    dates = pd.to_datetime({"year": years_rep, "month": months, "day": 1})

    enso_ts = pd.Series(enso_vals, index=dates, name="ENSO")
    enso_da = xr.DataArray(
        enso_ts.values, coords={"time": enso_ts.index}, dims="time", name="ENSO"
    )

    # drop duplicate months
    enso_da = enso_da.sel(time=~enso_da["time"].to_index().duplicated())
    precip = precip_ds.sel(time=slice(TIME_START, TIME_END))
    precip = precip.sel(time=~precip["time"].to_index().duplicated())

    # reindex ENSO on precip time
    enso_aligned = enso_da.interp(time=precip.time)

    # apply common mask
    mask = ~np.isnan(enso_aligned)
    enso_valid = enso_aligned.where(mask)
    precip_valid = precip.where(mask)

    corr = xr.corr(precip_valid[precip_var], enso_valid, dim="time")
    ds_out = xr.Dataset({"enso_index": corr})
    ds_out = finalize_geospatial(ds_out)
    return ds_out


def asc_to_zarr(asc_file, out_zarr):
    """Read .asc grid with rasterio and save as zarr."""
    print(f"Converting ASC to Zarr: {asc_file}")
    with rasterio.open(asc_file) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        ncols, nrows = src.width, src.height
        cellsize = transform[0]

    data = np.where(data == nodata, np.nan, data)

    x_coords = np.arange(transform[2], transform[2] + ncols * cellsize, cellsize)
    y_coords = np.arange(transform[5], transform[5] + nrows * -cellsize, -cellsize)

    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": y_coords, "lon": x_coords},
        name="irrigated_area",
        attrs={"nodata_value": nodata, "cellsize": cellsize, "source": "FAO GMIA v5 AAI%AEI"},
    )
    ds = xr.Dataset({"irrigated_area": da})
    ds = finalize_geospatial(ds)
    ds.to_zarr(out_zarr, mode="w")
    return ds


def interpolate_to_quarter_degree(ds, lat_name = "lat", lon_name = "lon"):

    lat_new = np.arange(90 - 0.25 / 2, -90, -0.25)
    lon_new = np.arange(-180 + 0.25 / 2, 180, 0.25)
    ds_interp = ds.interp({lat_name: lat_new, lon_name: lon_new}, method="linear")
    ds_interp = ds_interp.assign_coords({lat_name: lat_new, lon_name: lon_new})
    ds_interp = finalize_geospatial(ds_interp)
    return ds_interp


def restructure_lulcc_geotiff_to_zarr(geotiff_path, out_zarr):
    """
    Write lulcc file to Zarr with variables 'category' and 'change'.
    Also fixes weird artifact region.
    """
    print(f"Restructuring LULCC GeoTIFF: {geotiff_path}")
    ds = xr.open_dataset(geotiff_path, engine="rasterio")

    # Bounding box for weird artifact fix (lat 65..85, lon -100..5) on band=1 only
    lat_name = "y"
    lon_name = "x"
    mask = (ds[lat_name] >= 65) & (ds[lat_name] <= 85) & (ds[lon_name] >= -100) & (ds[lon_name] <= 5)

    band0 = ds.band_data.isel(band=0)  # category
    band1 = ds.band_data.isel(band=1)  # change
    band1_masked = band1.where(~mask, 0)

    # Rename coords and drop 'band' dim
    ds_ren = ds.rename({"x": "lon", "y": "lat"})
    category = band0.drop_vars("band")
    change = band1_masked.drop_vars("band")

    out = xr.Dataset({"category": category, "change": change})
    out = finalize_geospatial(out)
    out.to_zarr(out_zarr, mode="w")
    return out


def calculate_aridity_index(out_zarr): 

    AI_MEAN_MAX = 100.0        

    def wrap_to_180(ds, lon_name="lon"):
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
        return ds

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
    pet = xr.open_dataset("/mnt/data/romi/data/et_pot/monthly_sum_epot_clean.zarr").sel(
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

    # Package as a Dataset with a single var 'aridity_index'
    ai_ds = xr.Dataset({"aridity_index": ai_annual*10})

    # Stats + trend on annual AI 
    def to_numeric_years(time_coord: xr.DataArray) -> xr.DataArray:
        """Convert time coordinate to years since start for linear trend."""
        time_in_days = (time_coord - time_coord.isel(time=0)) / np.timedelta64(1, "D")
        return time_in_days / 365.25

    def compute_stats_and_trend(ds: xr.Dataset, var: str) -> xr.Dataset:
        """Compute mean, std, and trend-per-year for ds[var] (time = years)."""
        if var not in ds:
            raise KeyError(f"Variable '{var}' not in dataset. Available: {list(ds.data_vars)}")

        da = ds[var]

        years = to_numeric_years(ds.time)
        da_numt = da.assign_coords(time=years)

        coeffs = da_numt.polyfit(dim="time", deg=1)
        slope = coeffs.polyfit_coefficients.sel(degree=1)  # ΔAI per year

        mean_ = da.mean(dim="time")
        std_  = da.std(dim="time")

        return xr.Dataset({"mean": mean_, "std": std_, "monthly_trend": slope})

    ai_stats = compute_stats_and_trend(ai_ds, "aridity_index")
    ai_stats.to_zarr(out_zarr, mode="w")

    return 



# Google Earth Engine bits

def init_ee():
    """Authenticate and initialize Earth Engine ( if needed)."""
    import ee
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    return ee


def run_gee_pdsi():
    ee = init_ee()
    ic = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(TIME_START, TIME_END).select("pdsi")

    def add_time_band(img):
        date = ee.Image.constant(img.date().difference(ee.Date("2000-01-01"), "day")).toFloat()
        return img.addBands(date)

    ic_with_time = ic.map(add_time_band)
    mean_img = ic.mean().rename("mean")
    std_img = ic.reduce(ee.Reducer.stdDev()).rename("std")
    trend_img = ic_with_time.reduce(ee.Reducer.linearFit()).select("scale").rename("trend")

    metrics = mean_img.addBands(std_img).addBands(trend_img).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="PalmerIndex", folder="GEE_exports",
        fileNamePrefix="PDSI", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: PDSI")


def run_gee_gpp():
    ee = init_ee()
    ic = ee.ImageCollection("MODIS/061/MOD17A2H").select("Gpp")

    def add_time_band(img):
        date = ee.Image.constant(img.date().difference(ee.Date("2000-01-01"), "month")).toFloat()
        return img.addBands(date)

    ic_with_time = ic.map(add_time_band)
    mean_img = ic.mean().rename("mean")
    std_img = ic.reduce(ee.Reducer.stdDev()).rename("std")
    trend_img = ic_with_time.reduce(ee.Reducer.linearFit()).select("scale").rename("trend")
    metrics = mean_img.addBands(std_img).addBands(trend_img).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="GrossPrimaryProductivity", folder="GEE_exports",
        fileNamePrefix="GPP", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: GPP")


def run_gee_tree_cover():
    ee = init_ee()
    ic = ee.ImageCollection("MODIS/061/MOD44B").select("Percent_Tree_Cover").filterDate(TIME_START, TIME_END)

    def add_time_band(img):
        date = ee.Image.constant(img.date().difference(ee.Date("2000-01-01"), "month")).toFloat()
        return img.addBands(date)

    ic_with_time = ic.map(add_time_band)
    mean_img = ic.mean().rename("mean")
    std_img = ic.reduce(ee.Reducer.stdDev()).rename("std")
    trend_img = ic_with_time.reduce(ee.Reducer.linearFit()).select("scale").rename("trend")
    metrics = mean_img.addBands(std_img).addBands(trend_img).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="TreeCover", folder="GEE_exports",
        fileNamePrefix="tree_cover", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: TreeCover")


def run_gee_non_tree_cover():
    ee = init_ee()
    ic = ee.ImageCollection("MODIS/061/MOD44B").select("Percent_NonTree_Vegetation").filterDate(TIME_START, TIME_END)

    def add_time_band(img):
        date = ee.Image.constant(img.date().difference(ee.Date("2000-01-01"), "month")).toFloat()
        return img.addBands(date)

    ic_with_time = ic.map(add_time_band)
    mean_img = ic.mean().rename("mean")
    std_img = ic.reduce(ee.Reducer.stdDev()).rename("std")
    trend_img = ic_with_time.reduce(ee.Reducer.linearFit()).select("scale").rename("trend")
    metrics = mean_img.addBands(std_img).addBands(trend_img).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="NonTreeCover", folder="GEE_exports",
        fileNamePrefix="non_tree_cover", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: NonTreeCover")


def run_gee_soil_texture():
    ee = init_ee()
    img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0")
    img_agg = img.reproject(crs="EPSG:4326", scale=25000).reduceResolution(reducer=ee.Reducer.mode(), bestEffort=True)
    out = img_agg.reproject(crs="EPSG:4326", scale=25000)
    task = ee.batch.Export.image.toDrive(
        image=out, description="Soil_Texture_Features", scale=25000,
        crs="EPSG:4326", fileFormat="GeoTIFF", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: Soil Texture")


def run_gee_elevation():
    ee = init_ee()
    img = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    task = ee.batch.Export.image.toDrive(
        image=img, description="Elevation", folder="GEE_exports",
        fileNamePrefix="elevation", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: Elevation")


def run_gee_croplands():
    ee = init_ee()
    img = ee.Image("USGS/GFSAD1000_V1").select("landcover")
    img_agg = img.reproject(crs="EPSG:4326", scale=25000).reduceResolution(reducer=ee.Reducer.mode(), bestEffort=True)
    task = ee.batch.Export.image.toDrive(
        image=img_agg, description="Croplands", folder="GEE_exports",
        fileNamePrefix="crop_cover", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: Croplands")


def run_gee_ndvi():
    ee = init_ee()
    ic = ee.ImageCollection("MODIS/061/MOD13C1").select("NDVI").filterDate(TIME_START, TIME_END)

    def add_time_band(img):
        date = ee.Image.constant(img.date().difference(ee.Date("2000-01-01"), "month")).toFloat()
        return img.addBands(date)

    ic_with_time = ic.map(add_time_band)
    mean_img = ic.mean().rename("mean")
    std_img = ic.reduce(ee.Reducer.stdDev()).rename("std")
    trend_img = ic_with_time.reduce(ee.Reducer.linearFit()).select("scale").rename("trend")
    metrics = mean_img.addBands(std_img).addBands(trend_img).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="NDVI", folder="GEE_exports",
        fileNamePrefix="NDVI", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: NDVI")


def run_gee_fire():
    ee = init_ee()
    fire_coll = ee.ImageCollection("NASA/VIIRS/002/VNP14A1").filterDate("2012-01-01", TIME_END).select("FireMask")

    def to_binary_fire(img):
        fire = img.eq(8).rename("fire")
        return fire.set("system:time_start", img.date().millis())

    binary_coll = fire_coll.map(to_binary_fire)
    years = ee.List.sequence(2012, 2023)

    def yearly_metrics(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, "year")
        ycoll = binary_coll.filterDate(start, end)
        fire_days = ycoll.sum().rename("fire_days")
        total_days = ycoll.reduce(ee.Reducer.count()).rename("total_days")
        fire_fraction = fire_days.divide(total_days).rename("avg_fire_fraction")
        year_band = ee.Image.constant(year).rename("year").toFloat()
        return fire_days.addBands(fire_fraction).addBands(year_band).set("year", year)

    yearly_stack = ee.ImageCollection(years.map(yearly_metrics))
    trend = yearly_stack.select(["year", "fire_days"]).reduce(ee.Reducer.linearFit()).select("scale").rename("fire_trend")
    avg_fire_days = yearly_stack.select("fire_days").mean().rename("avg_fire_days")
    avg_fire_fraction = yearly_stack.select("avg_fire_fraction").mean().rename("avg_fire_fraction")

    metrics = avg_fire_days.addBands(avg_fire_fraction).addBands(trend).reproject(crs="EPSG:4326", scale=25000)

    task = ee.batch.Export.image.toDrive(
        image=metrics, description="Fire", folder="GEE_exports",
        fileNamePrefix="Fire", scale=25000, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    print("Started GEE export: Fire")



# Main


def main():
    # ERA5 temperature
    if RUN_ERA5_T2M:
        print("Processing ERA5 2m Temperature")
        ds_temp = open_grib_files(PATH_T2M)
        process_and_datasets(ds_temp, "t2m", str(OUT_DIR_FINAL / "driver_temperature.zarr"))

    # ERA5 precipitation
    if RUN_ERA5_PRECIP:
        print("Processing ERA5 Precipitation")
        ds_tp = open_grib_files(PATH_TP)
        process_and_datasets(ds_tp, "tp", str(OUT_DIR_FINAL / "driver_precipitation.zarr"))

    # ERA5 soil moisture 
    if RUN_ERA5_SOIL_MOISTURE:
        print("Processing ERA5 Soil Moisture (Layer 1)")
        ds_sm = open_grib_files(PATH_SWVL1)
        process_and_datasets(ds_sm, "swvl1", str(OUT_DIR_FINAL / "driver_soil_moisture.zarr"))

    # ERA5 evaporation from bare soil (which is just 'transpiration from plants but is mislabeled')
    if RUN_ERA5_TRANSPIRATION:
        print("Processing ERA5 Evaporation from Bare Soil")
        ds_evabs = open_grib_files(PATH_EVABS)
        process_and_datasets(ds_evabs, "evabs", str(OUT_DIR_FINAL / "driver_transpiration.zarr"))

    # ERA5 wind: u10/v10 to wind speed + direction
    if RUN_WIND:
        print("Processing ERA5 10m wind speed/direction")
        ds_u = open_grib_files(PATH_U10)
        ds_v = open_grib_files(PATH_V10)
        ds_u, ds_v = xr.align(ds_u, ds_v, join="inner")

        u = ds_u["u10"]
        v = ds_v["v10"]
        wind_speed = np.sqrt(u**2 + v**2)
        wind_dir = (np.degrees(np.arctan2(u, v)) + 180) % 360  # 0–360 instead of -180-180

        ws_ds = xr.Dataset({"wind_speed": wind_speed})
        wd_ds = xr.Dataset({"wind_dir": wind_dir})

        process_and_datasets(ws_ds, "wind_speed", str(OUT_DIR_FINAL / "driver_wind_speed.zarr"))
        process_and_datasets(wd_ds, "wind_dir", str(OUT_DIR_FINAL / "driver_wind_dir.zarr"))

    # PET processing (daily NetCDFs to monthly concatenation to stats + trend)
    if RUN_PET and PET_FILES:
        print("Processing PET (daily NetCDFs -> monthly stats)")
        with dask.config.set(scheduler="threads"):
            process_concatenate_pet(PET_FILES, "pet", str(OUT_DIR / "driver_pet.zarr"))

    if RUN_ARIDITY: 
        calculate_aridity_index(out_zarr = str(OUT_DIR_FINAL / "driver_aridity.zarr"))

    # Boundary layer height
    if RUN_ERA5_T2M: 
        print("Processing ERA5 Boundary Layer Height")
        ds_blh = open_grib_files(PATH_BLH)
        process_and_datasets(ds_blh, "blh", str(OUT_DIR_FINAL / "driver_boundary_layer_height.zarr"))

    # CAPE
    if RUN_ERA5_T2M:
        print("Processing ERA5 CAPE")
        ds_cape = open_grib_files(PATH_CAPE)
        process_and_datasets(ds_cape, "cape", str(OUT_DIR_FINAL / "driver_cape.zarr"))

    # Albedo (forecast)
    if RUN_ERA5_T2M:
        print("Processing ERA5 Forecast Albedo")
        ds_albedo = open_grib_files(PATH_ALBEDO)
        process_and_datasets(ds_albedo, "fal", str(OUT_DIR_FINAL / "driver_albedo.zarr"))

    # Groundwater table depth
    if RUN_GROUNDWATER:
        print("Processing Groundwater WTD datasets")
        processed = [process_groundwater_file(fp, variable="WTD") for fp in GW_FILES]
        # Save individual processed datasets and then harmonize + merge average
        out_paths = [fp.replace("monthlymeans.nc", "processed.zarr") for fp in GW_FILES]
        for ds, out_fp in zip(processed, out_paths):
            ds.to_zarr(out_fp, mode="w")
            print(f"Saved: {out_fp}")

        print("Harmonizing groundwater grids")
        datasets = [xr.open_dataset(fp) for fp in out_paths]
        harmonized = harmonize_grid(datasets)
        merged = xr.concat(harmonized, dim="dataset", join="outer").mean(dim="dataset")
        merged = merged.rename({"monthly_trend": "trend"})
        merged = finalize_geospatial(merged)
        merged.to_zarr(GW_OUT, mode="w")
        print(f"Saved merged groundwater dataset: {GW_OUT}")

    # ENSO correlation with precipitation
    if RUN_ENSO_CORRELATION:
        print("Computing ENSO correlation with ERA5 precipitation")
        ds_tp = open_grib_files(PATH_TP)
        ds_corr = enso_precip_correlation(ENSO_CSV, ds_tp, precip_var="tp")
        out = str(OUT_DIR_FINAL / "driver_enso.zarr")
        print(f"Saving ENSO correlation to: {out}")
        ds_corr.to_zarr(out, mode="w")

    # Irrigated areas
    if RUN_IRRIGATED_AREA:
        print("Processing irrigated area (ASC -> Zarr -> 0.25°)")
        ds_irrig = asc_to_zarr(GMIA_ASC, GMIA_ZARR)
        ds_irrig_025 = interpolate_to_quarter_degree(ds_irrig)
        ds_irrig_025.to_zarr(GMIA_ZARR, mode="w")
        print(f"Saved irrigated areas at 0.25°: {GMIA_ZARR}")

    # Land cover 
    if RUN_LULCC_RESTRUCTURE:
        print("Restructuring Land Cover features GeoTIFF -> Zarr")
        ds_lulcc = restructure_lulcc_geotiff_to_zarr(LULCC_TIF, LULCC_ZARR)
        ds_lulcc_interp = interpolate_to_quarter_degree(ds_lulcc)
        ds_lulcc_interp.to_zarr(LULCC_ZARR, mode="w")
        print(f"Saved land cover features to: {LULCC_ZARR}")


    # Optional GEE exports
    if RUN_GEE_PDSI:
        run_gee_pdsi()
    if RUN_GEE_GPP:
        run_gee_gpp()
    if RUN_GEE_TREE_COVER:
        run_gee_tree_cover()
    if RUN_GEE_NON_TREE_COVER:
        run_gee_non_tree_cover()
    if RUN_GEE_SOIL_TEXTURE:
        run_gee_soil_texture()
    if RUN_GEE_ELEVATION:
        run_gee_elevation()
    if RUN_GEE_CROPLANDS:
        run_gee_croplands()
    if RUN_GEE_NDVI:
        run_gee_ndvi()
    if RUN_GEE_FIRE:
        run_gee_fire()

    print("Done! Phew")


if __name__ == "__main__":

    main()