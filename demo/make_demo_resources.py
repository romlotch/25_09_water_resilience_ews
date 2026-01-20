# demo/make_demo_resources.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

try:
    import rioxarray  # noqa: F401
    HAS_RIO = True
except Exception:
    HAS_RIO = False


REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"
DATA_ZARR = DEMO / "data" / "demo_sm.zarr"

RESOURCES = DEMO / "resources"
BIOMES_DIR = RESOURCES / "biomes"
MASKS_DIR = RESOURCES / "masks"


def _open_demo_ds() -> xr.Dataset:
    if not DATA_ZARR.exists():
        raise FileNotFoundError(
            f"Missing {DATA_ZARR}. Run demo/make_demo_data.py first."
        )
    return xr.open_dataset(DATA_ZARR, engine="zarr")


def make_landsea_mask(ds: xr.Dataset) -> None:
    """
    Create a trivial land-sea mask on the same lat/lon grid.
    We mark all pixels as land except a simple 'ocean' stripe for demo maps.
    """
    lat = ds["lat"]
    lon = ds["lon"]

    # Start with all land
    lsm = xr.DataArray(
        np.ones((lat.size, lon.size), dtype="float32"),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="lsm",
        attrs={"long_name": "Demo land-sea mask", "units": "1"},
    )

    # Make a fake ocean stripe 
    ocean = lon < float(lon.min() + (lon.max() - lon.min()) * 0.25)
    lsm.loc[{"lon": lon[ocean.values]}] = 0.0

    mask_ds = xr.Dataset({"lsm": lsm})

    # Optional CRS metadata 
    if HAS_RIO:
        mask_ds = mask_ds.rio.write_crs("EPSG:4326", inplace=False)

    out_nc = RESOURCES / "landsea_mask_demo.nc"
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    mask_ds.to_netcdf(out_nc)
    print(f"Wrote {out_nc}")


def make_biome_shapefile(ds: xr.Dataset) -> None:
    """
    Create a minimal shapefile with 2 polygons covering the demo grid.
    Includes a few common-ish fields that biome scripts often look for.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values
    minx, maxx = float(lon.min()), float(lon.max())
    miny, maxy = float(lat.min()), float(lat.max())
    midx = (minx + maxx) / 2.0

    # Two rectangular "biomes"
    geom_west = box(minx, miny, midx, maxy)
    geom_east = box(midx, miny, maxx, maxy)

    gdf = gpd.GeoDataFrame(
        {
            # Attribute names
            "ECO_NAME": ["Demo Biome West", "Demo Biome East"],
            "BIOME_NAME": ["Biome_W", "Biome_E"],
            "BIOME": [1, 2],
            "REALM": ["DEMO", "DEMO"],
            "ECO_ID": [1, 2],
        },
        geometry=[geom_west, geom_east],
        crs="EPSG:4326",
    )

    BIOMES_DIR.mkdir(parents=True, exist_ok=True)
    out_shp = BIOMES_DIR / "tnc_terr_ecoregions.shp"
    gdf.to_file(out_shp)
    print(f"Wrote {out_shp}")


def make_optional_masks(ds: xr.Dataset) -> None:
    """
    Optional boolean masks as tiny Zarr stores. Only needed if scripts open them.
    """
    lat = ds["lat"]
    lon = ds["lon"]
    nlat, nlon = lat.size, lon.size

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    def write_mask(name: str, arr: np.ndarray) -> None:
        da = xr.DataArray(
            arr.astype("uint8"),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name="mask",
        )
        out = MASKS_DIR / f"{name}.zarr"
        xr.Dataset({name: da, "mask": da}).to_zarr(out, mode="w")
        print(f"Wrote {out}")

    # Simple patterns 
    urban = np.zeros((nlat, nlon), dtype="uint8")
    urban[: max(1, nlat // 6), : max(1, nlon // 6)] = 1

    crop = np.zeros((nlat, nlon), dtype="uint8")
    crop[max(1, nlat // 3) : max(2, 2 * nlat // 3), :] = 1

    irrig = np.zeros((nlat, nlon), dtype="uint8")
    irrig[:, max(1, nlon // 3) : max(2, 2 * nlon // 3)] = 1

    write_mask("urban_mask", urban)
    write_mask("crop_mask", crop)
    write_mask("irrigated_areas", irrig)


if __name__ == "__main__":
    ds = _open_demo_ds()
    RESOURCES.mkdir(parents=True, exist_ok=True)
    make_landsea_mask(ds)
    make_biome_shapefile(ds)
    make_optional_masks(ds)
