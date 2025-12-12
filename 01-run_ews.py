

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.seasonal import STL
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import gc

# Patch psutil bug 
import builtins, psutil._common
def _raise_from(err): raise err
psutil._common.raise_from = _raise_from
def _open_binary(path, mode='rb', **kwargs): return builtins.open(path, mode, **kwargs)
psutil._common.open_binary = _open_binary


""" 
Run EWS on input time series in pure numpy (faster in parallel than with xarray/dask). 
First creates tiles and saves them to a folder the current working directory.
Then runs the EWS on each tile and saves them to a separate working directory. 

Inputs:
    --dataset: path to dataset (format that can be read by xarray)
    --variable: name of the variable on which to run the EWS
    --freq: specify the resampling frequency (i.e. 'W' or keep daily)
    --workers: specify number of workers

Saves 
 
"""



def open_source_ds(path):
   
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup")) or
        os.path.exists(os.path.join(path, ".zmetadata"))
    ):
        return xr.open_zarr(path)
    else:
        return xr.open_dataset(path)


def standardize_lat_lon(ds):
    """Rename dims to lat, lon, time, set projection and centre on 0 instead of 180"""
    
    rename = {}
    if "latitude" in ds.dims:  rename["latitude"] = "lat"
    if "longitude" in ds.dims: rename["longitude"] = "lon"
    if "datetime" in ds.dims:  rename["datetime"] = "time"

    ds = ds.rio.write_crs("EPSG:4326")
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    return ds


def _infer_weekly_agg(variable):
    """Choose SUM for precipitation variables, otherwise use the MEAN."""
    v = variable.lower()
    precip_aliases = {"precip","pr","precipitation","tp","rain","rainfall"}
    return "sum" if any(a in v for a in precip_aliases) else "mean"


def preprocess(ds, variable, freq: str = "D", t0: str = "2000-01-01", t1: str = "2023-12-31"):
    """
    Standardize lon/lat, slice time to [t0, t1], and resample to weekly.
    Weekly aggregation is SUM for precip-like variables, otherwise MEAN.
    """
    ds = standardize_lat_lon(ds)
    if "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' coordinate.")
    ds = ds.sel(time=slice(np.datetime64(t0), np.datetime64(t1)))

    if isinstance(freq, str) and freq.upper().startswith("W"):
        how = _infer_weekly_agg(variable)
        if how == "sum":
            ds = ds.resample(time=freq).sum(keep_attrs=True, skipna=True)
        else:
            ds = ds.resample(time=freq).mean(keep_attrs=True, skipna=True)
    return ds


def _write_tile_from_path(dataset_path, variable, tile_path, lat_slice, lon_slice, freq: str):
    """Slice out a tile and write it."""

    ds = open_source_ds(dataset_path)

    ds = preprocess(ds, variable=variable, freq=freq, t0="2000-01-01", t1="2023-12-31")

    tile = ds[[variable]].isel(lat=lat_slice, lon=lon_slice)
    tile = tile.chunk({"time": -1, "lat": -1, "lon": -1})
    tile.to_zarr(tile_path, mode="w")
    ds.close()


def create_tiles(dataset_path, variable, out_dir, tile_size: int = 50, workers: int = 4, freq: str = "D"):
    """
    Parallel tile creation: slice dataset into spatial tiles and write each tile Zarr.
    """
    # read metadata
    ds = open_source_ds(dataset_path)

    # land mask precip so it doesn't take forever
    if variable == 'precip': 

        mask_ds = xr.open_dataset("/mnt/data/romi/data/landsea_mask.grib", engine="cfgrib")

        mask_ds = mask_ds.rio.write_crs("EPSG:4326")
        mask_ds = mask_ds.assign_coords(longitude=(((mask_ds.longitude + 180) % 360) - 180)).sortby("longitude")
        mask_ds.rio.set_spatial_dims("longitude", "latitude", inplace=True)

        # Interpolate to ds grid 
        mask_on_ds = mask_ds.interp(
            longitude=ds["lon"],  
            latitude=ds["lat"],
            method="nearest",
        )

        # Threshold and apply
        land = mask_on_ds["lsm"] > 0.7
        ds = ds.where(land)

    else: 

        ds = ds

    ds = standardize_lat_lon(ds)
    lat_dim, lon_dim = ds.sizes["lat"], ds.sizes["lon"]
    ds.close()

    # prepare jobs
    n_lat = (lat_dim + tile_size - 1) // tile_size
    n_lon = (lon_dim + tile_size - 1) // tile_size
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for i in range(n_lat):
        for j in range(n_lon):
            path = os.path.join(out_dir, f"tile_{i}_{j}.zarr")
            if not os.path.exists(path):
                ls = slice(i*tile_size, min((i+1)*tile_size, lat_dim))
                lo = slice(j*tile_size, min((j+1)*tile_size, lon_dim))
                jobs.append((dataset_path, variable, path, ls, lo))

    if not jobs:
        print(f"[TILING] no new tiles in {out_dir}")
        return

    print(f"[TILING] Writing {len(jobs)} tiles with {workers} workers → {out_dir}/")
    with ProcessPoolExecutor(max_workers=workers) as exe:

        futures = { exe.submit(_write_tile_from_path, *job, freq): job for job in jobs }
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[TILING-ERROR] {job[2]}: {e}", file=sys.stderr)
            pct = i/len(jobs)*100
            print(f"\r[TILING] {i}/{len(jobs)} ({pct:.0f}%)", end="")

    print("\n[TILING] Done.")


def _infer_stl_period(times, freq_hint):
    """52 for weekly, else 365 for dailye."""
    if isinstance(freq_hint, str) and freq_hint.upper().startswith("W"):
        return 52
    # fall back: infer from median step
    t = pd.to_datetime(times)
    if len(t) >= 2:
        step = (t[1:] - t[:-1]).median().days
        if step >= 5:  # weekly or coarser
            return 52
    return 365


def remove_multiple_trends_STL(values, times, period):
    if np.all(np.isnan(values)):
        return np.full_like(values, np.nan)
    ts = pd.Series(values, index=pd.to_datetime(times))
    missing = ts.isna()
    filled = ts.fillna(ts.mean())

    # keep seasonal/trend windows roughly proportional to period
    seasonal = max(7, min( int(round(period/4)), 61))
    trend    = max(  2*period-1, 101)
    resid = STL(filled, period=period, seasonal=seasonal, trend=trend, robust=True).fit().resid
    resid[missing] = np.nan

    # old: resid = STL(filled, period=52, seasonal=13, trend=201, robust=True).fit().resid
    # new infers period and seasonality 
    
    return resid.values

def rolling_ews(ts, window):

    """
    AC1, STD, SKEW, KURT, FD in a centered rolling window.
    """
    n = len(ts); half = window // 2
    out_ac1 = np.full(n, np.nan)
    out_std = np.full(n, np.nan)
    out_skew = np.full(n, np.nan)
    out_kurt = np.full(n, np.nan)
    out_fd  = np.full(n, np.nan)

    for k in range(half, n - half):
        w = ts[k-half:k+half+1]
        msk = ~np.isnan(w)
        n_valid = int(msk.sum())

        if n_valid != len(w):
                continue

        m = np.nanmean(w)
        v = np.nanvar(w)
        s = np.sqrt(v)
        out_std[k] = s
    
        num = np.sum((w[1:] - m) * (w[:-1] - m))
        out_ac1[k] = (num / n_valid) / v

        wn = (w - m) / s
        out_skew[k] = np.mean(wn**3)
        out_kurt[k] = np.mean(wn**4)

        g1 = np.log(np.sum(np.abs(w[1:] - w[:-1])) / (2 * (n_valid - 1))) if n_valid > 1 else np.nan
        g2 = np.log(np.sum(np.abs(w[2:] - w[:-2])) / (2 * (n_valid - 2))) if n_valid > 2 else np.nan
        if np.isfinite(g1) and np.isfinite(g2):
            x = np.log([1.0, 2.0])
            xbar = x.mean()
            x_corr = x - xbar
            fd_raw = 2.0 - (x_corr[0]*g1 + x_corr[1]*g2) / np.sum((x - xbar)**2)
            out_fd[k] = np.clip(fd_raw, 1.0, 2.0)

    return out_ac1, out_std, out_skew, out_kurt, out_fd

def compute_tile_outputs(raw,times,lat, lon, variable, window, freq_hint = None):
    """
    Given a (time,lat,lon) raw np array, either return all-NaN placeholders
    or compute the detrended series, EWS stats, and deltas.
    Returns (data_vars, coords).
    """
    nt, nlat, nlon = raw.shape
    data_vars = {}

    # placeholder if entirely NaN
    if np.all(np.isnan(raw)):

        shape3 = (nt, nlat, nlon)
        data_vars[variable] = (("time","lat","lon"), np.full(shape3, np.nan))
        data_vars[f"{variable}_detrended"] = (("time","lat","lon"), np.full(shape3, np.nan))

        for stat in ("ac1","std","skew","kurt","fd"):
            data_vars[f"{variable}_{stat}"]       = (("time","lat","lon"), np.full(shape3, np.nan))
            data_vars[f"{variable}_delta_{stat}"] = (("lat","lon"),            np.full((nlat,nlon), np.nan))

        coords = {"time": times, "lat": lat, "lon": lon}
        return data_vars, coords

    # STL detrend
    period = _infer_stl_period(times, freq_hint)
    flat  = raw.reshape(nt,-1)
    resid = np.empty_like(flat)
    for i in range(flat.shape[1]):
        resid[:,i] = remove_multiple_trends_STL(flat[:,i], times, period)
    resid3 = resid.reshape(nt,nlat,nlon)

    # First difference
    detr = np.empty_like(resid3)
    detr[0] = np.nan
    detr[1:] = resid3[1:] - resid3[:-1]

    # Rolling EWS
    ac1_list = []; std_list = []; skew_list = []; kurt_list = []; fd_list = []
    for idx in range(nlat*nlon):
        series = detr[:, idx//nlon, idx%nlon]
        a,s,k,ku,f = rolling_ews(series, window)
        ac1_list.append(a); std_list.append(s)
        skew_list.append(k); kurt_list.append(ku)
        fd_list.append(f)

    ac1 = np.vstack(ac1_list).T.reshape(nt,nlat,nlon)
    std = np.vstack(std_list).T.reshape(nt,nlat,nlon)
    skew = np.vstack(skew_list).T.reshape(nt,nlat,nlon)
    kurt = np.vstack(kurt_list).T.reshape(nt,nlat,nlon)
    fd = np.vstack(fd_list).T.reshape(nt,nlat,nlon)

    # Deltas
    for name, arr in [("ac1",ac1),("std",std),("skew",skew),
                      ("kurt",kurt),("fd",fd)]:
        
        with np.errstate(invalid="ignore", divide="ignore"):
            
            mx  = np.nanmax(arr, axis=0)
            mn  = np.nanmin(arr, axis=0)

            # build empty index arrays
            imx = np.zeros_like(mx, dtype=int)
            imn = np.zeros_like(mx, dtype=int)

            valid = ~np.isnan(mx)
            # only compute argmax/argmin where there's at least one real value
            for ii in range(mx.shape[0]):
                for jj in range(mx.shape[1]):
                    if valid[ii,jj]:
                        imx[ii,jj] = np.nanargmax(arr[:,ii,jj])
                        imn[ii,jj] = np.nanargmin(arr[:,ii,jj])

            sign  = np.where(imn > imx, -1, 1)
            delta = sign * (mx - mn)

        data_vars[f"{variable}_delta_{name}"] = (("lat","lon"), delta)
            

    # assemble it all into two dicts 
    data_vars[variable]             = (("time","lat","lon"), raw)
    data_vars[f"{variable}_detrended"] = (("time","lat","lon"), detr)
    data_vars[f"{variable}_ac1"]    = (("time","lat","lon"), ac1)
    data_vars[f"{variable}_std"]    = (("time","lat","lon"), std)
    data_vars[f"{variable}_skew"]   = (("time","lat","lon"), skew)
    data_vars[f"{variable}_kurt"]   = (("time","lat","lon"), kurt)
    data_vars[f"{variable}_fd"]     = (("time","lat","lon"), fd)
    

    coords = {"time": times, "lat": lat, "lon": lon}
    return data_vars, coords

def process_tile(tile_path, variable, window, tiles_dir, proc_dir, freq_hint=None):
    # load & prep 
    ds   = open_source_ds(tile_path)
    da   = ds[variable].transpose("time","lat","lon")
    raw  = da.values
    times= da["time"].values
    lat  = da["lat"].values
    lon  = da["lon"].values
    ds.close()

    # build out_path
    rel      = os.path.relpath(tile_path, tiles_dir)
    out_path = os.path.join(proc_dir, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # compute
    data_vars, coords = compute_tile_outputs(raw, times, lat, lon, variable, window, freq_hint=freq_hint)
    print(f"   [DEBUG] vars -> {list(data_vars)}")

    # write
    out_ds = xr.Dataset(data_vars, coords=coords)
    out_ds.to_zarr(out_path, mode="w")
    print(f"   [DEBUG] wrote -> {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   required=True)
    p.add_argument('--variable',  required=True)
    p.add_argument('--freq')
    p.add_argument('--window_years', type=int, default=5)
    p.add_argument('--out',         default=None)
    p.add_argument('--workers',     type=int, default=None)
    args = p.parse_args()

    workers = args.workers or os.cpu_count() or 1
    suffix  = args.out or args.variable
    tiles   = f"tiles_{suffix}"
    processed = f"processed_tiles_{suffix}"

    print(f'Tiles dir: {tiles}')
    print(f'Processed dir: {processed}')

    print(f"[MAIN] Tiling with {workers} workers")
    create_tiles(args.dataset, args.variable, tiles,
                 tile_size=50, workers=workers, freq=args.freq)

    print(f"[MAIN] Processing with {workers} workers")
    os.makedirs(processed, exist_ok=True)

    tile_list = sorted(glob.glob(os.path.join(tiles,"tile_*.zarr")))
    freq_str = args.freq or "D"
    window = (52 if isinstance(freq_str, str) and freq_str.upper().startswith("W") else 365) * args.window_years

    fn = partial(process_tile,
                 variable=args.variable,
                 window=window,
                 tiles_dir=tiles,
                 proc_dir=processed,
                 freq_hint=freq_str)

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = { exe.submit(fn, tp): tp for tp in tile_list }
        done = 0
        for fut in as_completed(futures):
            tp = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] {tp}: {e}", file=sys.stderr)
            done+=1
            pct = done/len(tile_list)*100
            print(f"\r[MAIN] {done}/{len(tile_list)} ({pct:.0f}%)", end="")
    print("\n[MAIN] All done.")


if __name__=='__main__':

    # python3 run_ews_final.py --dataset /mnt/data/romi/data/PERSIANN_CDR_daily/persiann_cdr.zarr --variable 'precip' --freq 'W' --workers 8
    # python3 run_ews_final.py --dataset /mnt/data/romi/data/esa_cci_sm_combined.zarr --variable 'sm' --freq 'W' --workers 8
    # python3 run_ews_final.py --dataset /mnt/data/romi/data/GLEAM_v4.2a/regridded/Et_GLEAM_v4.2a_025.nc --variable 'Et' --freq 'W' --workers 8

    ## Run it on datasets before breakpoints 

    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_Et_final/Et_cp_masked_stc.zarr' --variable 'Et' --freq 'W' --out Et_breakpoints
    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_sm_final/sm_cp_masked_stc.zarr' --variable 'sm' --freq 'W' --out sm_breakpoints
    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_precip_final/precip_cp_masked_stc.zarr' --variable 'precip' --freq 'W' --out precip_breakpoints

    ## Run it on the negatives with the fake breakpoints 

    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_Et_final/Et_cp_masked_stc_neg.zarr' --variable 'Et' --freq 'W' --out Et_breakpoints_neg
    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_sm_final/sm_cp_masked_stc_neg.zarr' --variable 'sm' --freq 'W' --out sm_breakpoints_neg
    # python3 01-run_ews.py --dataset  '/mnt/data/romi/output/paper_1/output_precip_final/precip_cp_masked_stc_neg.zarr' --variable 'precip' --freq 'W' --out precip_breakpoints_neg

    main()
