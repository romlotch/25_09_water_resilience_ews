import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.seasonal import STL


""" 
Run a simple sensitivity analysis that tests effect of temporal aggregation, 
window size, and some detrending parameters (robust STL and including diff or not)
on a tile.

Inputs: 
    --dataset: ews output to run the sensitivity on (uses the raw data variable that is still in the file)
    --variable: variable name
    --outdir: specify the output dir
    --i0, i1, j0, j1: indices to slice a tile 
    --eps: optionally specify tolerance for when delta is 0 

python3 01c-sensitivity.py \
            --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr \
            --variable precip \
            --outdir /mnt/data/romi/output/paper_1/output_precip_final \
            --i0 270 --i1 320 --j0 470 --j1 520: indices to slice out the tile (currently in SA)
            
            
"""

# --- helpers ---

def open_source_ds(path):
    if os.path.isdir(path) and (os.path.exists(os.path.join(path, ".zgroup"))
                                or os.path.exists(os.path.join(path, ".zmetadata"))):
        return xr.open_zarr(path)
    return xr.open_dataset(path)

def standardize_lat_lon(ds):
    rename = {}
    if "latitude"  in ds.dims: rename["latitude"]  = "lat"
    if "longitude" in ds.dims: rename["longitude"] = "lon"
    if "datetime"  in ds.dims: rename["datetime"]  = "time"
    if rename:
        ds = ds.rename(rename)
    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must have lon/lat dims.")
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds

def _infer_agg(variable: str):
    v = variable.lower()
    return "sum" if any(a in v for a in {"precip","pr","precipitation","tp","rain"}) else "mean"

def resample_to(ds, variable, freq: str | None):
    """None or 'D' -> keep; 'W' -> weekly; 'MS' -> monthly start."""
    if (freq is None) or (str(freq).upper() == "D"):
        return ds
    how = _infer_agg(variable)
    if str(freq).upper().startswith("W"):
        return (ds.resample(time=freq).sum(keep_attrs=True, skipna=True)
                if how=="sum" else
                ds.resample(time=freq).mean(keep_attrs=True, skipna=True))
    if str(freq).upper() in ("M","MS"):
        return (ds.resample(time="MS").sum(keep_attrs=True, skipna=True)
                if how=="sum" else
                ds.resample(time="MS").mean(keep_attrs=True, skipna=True))
    raise ValueError(f"Unsupported freq {freq}. Use D, W, or MS.")

def steps_per_year_from_times(times):
    t = pd.to_datetime(times)
    span_years = (t[-1] - t[0]).days / 365.2425
    # robust to irregular sampling
    return max(3, int(round(len(t) / span_years)))

def stl_period_for(times):
    steps = steps_per_year_from_times(times)
    # for weekly data steps = 52; monthly = 12; daily = 365 approx
    return steps

def make_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

# --- STL detrend ---

def stl_residual(values, times, period, robust=True, fill="mean"):
    if np.all(np.isnan(values)):
        return np.full_like(values, np.nan)
    ts = pd.Series(values, index=pd.to_datetime(times))

    if fill == "mean":
        filled = ts.fillna(ts.mean())
    elif fill == "linear":
        filled = ts.interpolate("time").fillna(ts.mean())
    else:
        raise ValueError("fill must be 'mean' or 'linear'")

    # windows proportional to period
    seasonal = max(7, min(int(round(period/4)), 61))
    trend    = max(2*period-1, 101)
    res = STL(filled, period=period, seasonal=seasonal, trend=trend, robust=robust).fit().resid
    res[ts.isna()] = np.nan
    return res.values

# --- rolling EWS ---

def rolling_ews_centered(ts, window):
    n = len(ts); half = window//2
    out = [np.full(n, np.nan) for _ in range(5)]  # ac1, sd, skew, kurt, fd
    for k in range(half, n-half):
        w = ts[k-half:k+half+1]
        if np.sum(~np.isnan(w)) > 2:
            m  = np.nanmean(w)
            v  = np.nanvar(w, ddof=1)
            s  = np.nanstd(w, ddof=1)
            out[1][k] = s
            out[0][k] = np.nansum((w[1:]-m)*(w[:-1]-m)) / (v*(len(w)-1))
            out[2][k] = np.nanmean(((w-m)/s)**3)
            out[3][k] = np.nanmean(((w-m)/s)**4)
            d1 = np.nansum(np.abs(w[1:]-w[:-1])) / (2*(len(w)-1))
            d2 = np.nansum(np.abs(w[2:]-w[:-2])) / (2*(len(w)-2))
            if d1>0 and d2>0:
                x   = np.log([1,2]); xb = x.mean()
                fdv = 2 - ((x[0]-xb)*np.log(d1) + (x[1]-xb)*np.log(d2)) / ((x-xb)**2).sum()
                out[4][k] = np.clip(fdv, 1, 2)
    return tuple(out)

def deltas_from_series(arr3, stat_name):
    """Signed delta = (max - min) with sign by order of occurrence."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.nanmax(arr3, axis=0)
        mn = np.nanmin(arr3, axis=0)
        imx = np.zeros_like(mx, dtype=int)
        imn = np.zeros_like(mx, dtype=int)
        valid = np.isfinite(mx)
        lt, ly, lx = arr3.shape
        for ii in range(ly):
            for jj in range(lx):
                if valid[ii, jj]:
                    s = arr3[:, ii, jj]
                    imx[ii, jj] = np.nanargmax(s)
                    imn[ii, jj] = np.nanargmin(s)
        sign  = np.where(imn > imx, -1, 1)
        delta = sign * (mx - mn)
    return delta

# --- sign-first metrics ---

def sign_map(delta_da, eps: float = 0.0):
    """Map delta to {-1,0,1} with tolerance eps."""
    return xr.where(delta_da > eps, 1,
           xr.where(delta_da < -eps, -1, 0)).astype(np.int8)

def sign_prevalence(delta_da, eps: float = 0.0):
    """Unweighted % up / % down / % neutral and dominance (%up - %down)."""
    valid = np.isfinite(delta_da)
    N = int(valid.sum())
    if N == 0:
        return {"pct_up": np.nan, "pct_down": np.nan, "pct_neutral": np.nan, "dom": np.nan}
    up   = float(((delta_da >  eps) & valid).sum()) / N * 100.0
    down = float(((delta_da < -eps) & valid).sum()) / N * 100.0
    neu  = 100.0 - up - down
    dom  = up - down
    return {"pct_up": up, "pct_down": down, "pct_neutral": neu, "dom": dom}

# --- core compute for a tile ---

def compute_ews_for_tile(raw3, times, freq, years_window,
                         do_diff=True, stl_robust=True, fill="mean"):
    """
    freq: 'D', 'W', 'MS'
    years_window: 2.5, 5, 10
    do_diff: first-difference before EWS?
    stl_robust: robust STL on/off
    """
    nt, nlat, nlon = raw3.shape
    period = stl_period_for(times)

    # STL residual per pixel
    flat = raw3.reshape(nt, -1)
    resid = np.empty_like(flat)
    for i in range(flat.shape[1]):
        resid[:, i] = stl_residual(flat[:, i], times, period=period, robust=stl_robust, fill=fill)
    resid3 = resid.reshape(nt, nlat, nlon)

    # First difference 
    if do_diff:
        detr = np.empty_like(resid3)
        detr[0] = np.nan
        detr[1:] = resid3[1:] - resid3[:-1]
    else:
        detr = resid3.copy()

    # Rolling window in steps
    steps = steps_per_year_from_times(times)
    win   = make_odd(int(round(years_window * steps)))
    if win >= len(times):
        win = make_odd(min(len(times) - 1, int(round(0.8 * len(times)))))

    ac1_list, sd_list, skew_list, kurt_list, fd_list = [], [], [], [], []
    for idx in range(nlat*nlon):
        series = detr[:, idx//nlon, idx % nlon]
        a, s, sk, ku, fd = rolling_ews_centered(series, win)
        ac1_list.append(a); sd_list.append(s); skew_list.append(sk); kurt_list.append(ku); fd_list.append(fd)
    ac1  = np.vstack(ac1_list).T.reshape(nt, nlat, nlon)
    sd   = np.vstack(sd_list).T.reshape(nt, nlat, nlon)
    skew = np.vstack(skew_list).T.reshape(nt, nlat, nlon)
    kurt = np.vstack(kurt_list).T.reshape(nt, nlat, nlon)
    fd   = np.vstack(fd_list).T.reshape(nt, nlat, nlon)

    # Deltas
    d_ac1  = deltas_from_series(ac1,  "ac1")
    d_sd   = deltas_from_series(sd,   "std")
    d_skew = deltas_from_series(skew, "skew")
    d_kurt = deltas_from_series(kurt, "kurt")
    d_fd   = deltas_from_series(fd,   "fd")

    data_vars = {
        "raw":              (("time","lat","lon"), raw3),
        "residual":         (("time","lat","lon"), resid3),
        "detrended":        (("time","lat","lon"), detr),
        "ac1":              (("time","lat","lon"), ac1),
        "std":              (("time","lat","lon"), sd),
        "skew":             (("time","lat","lon"), skew),
        "kurt":             (("time","lat","lon"), kurt),
        "fd":               (("time","lat","lon"), fd),
        "delta_ac1":        (("lat","lon"), d_ac1),
        "delta_std":        (("lat","lon"), d_sd),
        "delta_skew":       (("lat","lon"), d_skew),
        "delta_kurt":       (("lat","lon"), d_kurt),
        "delta_fd":         (("lat","lon"), d_fd),
    }
    coords = {"time": times, "lat": np.arange(nlat), "lon": np.arange(nlon)}
    return xr.Dataset(data_vars, coords=coords)

# --- tile ---

def load_tile(dataset_path, variable, freq_resample, i0, i1, j0, j1,
              t0="2000-01-01", t1="2023-12-31"):
    ds = open_source_ds(dataset_path)
    ds = standardize_lat_lon(ds)
    if "time" not in ds:
        raise ValueError("Dataset has no time coordinate.")
    ds = ds.sel(time=slice(np.datetime64(t0), np.datetime64(t1)))

    # resample
    ds_r = resample_to(ds, variable, freq_resample)

    # slice tile (by index)
    da = ds_r[variable].isel(lat=slice(i0, i1), lon=slice(j0, j1)).transpose("time","lat","lon")
    raw3  = da.values
    times = pd.to_datetime(da["time"].values)
    ds.close()
    return raw3, times

def safe_name(cfg):
    tag = f"freq{cfg['freq']}_win{cfg['win_years']}y_diff{int(cfg['diff'])}_stl{'R' if cfg['stl_robust'] else 'N'}"
    return tag.replace(".","p")

# --- sensitivity runner--- 

def summarize_signs(ds_out, eps=0.0, prefix=""):
    out = {}
    for key, label in [("delta_ac1","AC1"), ("delta_std","SD"),
                       ("delta_skew","Skew"), ("delta_kurt","Kurt"), ("delta_fd","FD")]:
        if key not in ds_out: 
            continue
        s = sign_prevalence(ds_out[key], eps=eps)
        out.update({f"{prefix}{label}_pct_up": s["pct_up"],
                    f"{prefix}{label}_pct_down": s["pct_down"],
                    f"{prefix}{label}_pct_neutral": s["pct_neutral"],
                    f"{prefix}{label}_dom": s["dom"]})
    return out

def run_sensitivity(dataset, variable, outdir,
                    i0=0, i1=50, j0=0, j1=50,
                    freqs=("D","W","MS"),
                    windows_years=(2.5, 5.0, 10.0),
                    diffs=(True, False),
                    stl_robust_opts=(True, False),
                    fill="mean",
                    eps_sign=0.0):
    os.makedirs(outdir, exist_ok=True)

    # grid of configs
    SENSITIVITIES = []
    for f in freqs:
        for wy in windows_years:
            for d in diffs:
                for r in stl_robust_opts:
                    SENSITIVITIES.append({"freq": f, "win_years": wy, "diff": d, "stl_robust": r})

    # Pre-load raw series for each frequency (avoid re-reading)
    cache = {}
    for f in freqs:
        raw3, times = load_tile(dataset, variable, f, i0, i1, j0, j1)
        cache[f] = (raw3, times)

    summaries = []
    for cfg in SENSITIVITIES:
        f   = cfg["freq"]
        wy  = cfg["win_years"]
        dff = cfg["diff"]
        rob = cfg["stl_robust"]

        raw3, times = cache[f]
        ds_out = compute_ews_for_tile(raw3, times, f, wy, do_diff=dff, stl_robust=rob, fill=fill)

        tag = safe_name(cfg)
        out_path = os.path.join(outdir, f"sensitivity_{variable}_{tag}.zarr")
        if os.path.exists(out_path):
            import shutil; shutil.rmtree(out_path)
        ds_out.to_zarr(out_path, mode="w", consolidated=True)

        # magnitudes (keep)
        row = {
            "freq": f, "win_years": wy, "diff": dff, "stl_robust": rob,
            "mean_abs_delta_ac1":  float(np.nanmean(np.abs(ds_out["delta_ac1"].values))),
            "mean_abs_delta_std":  float(np.nanmean(np.abs(ds_out["delta_std"].values))),
            "mean_abs_delta_skew": float(np.nanmean(np.abs(ds_out["delta_skew"].values))),
            "mean_abs_delta_kurt": float(np.nanmean(np.abs(ds_out["delta_kurt"].values))),
            "mean_abs_delta_fd":   float(np.nanmean(np.abs(ds_out["delta_fd"].values))),
        }
        # sign-first metrics (new)
        row.update(summarize_signs(ds_out, eps=eps_sign))
        summaries.append(row)
        print(f"[OK] wrote {out_path}")

    # Save CSV
    df = pd.DataFrame(summaries)
    csv_path = os.path.join(outdir, f"{variable}_sensitivity_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DONE] Summary saved → {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Sensitivity analysis on a single tile (sign-first + magnitudes).")
    ap.add_argument("--dataset",  required=True, help="Path to full dataset (zarr or netcdf)")
    ap.add_argument("--variable", required=True, help="Variable name (e.g., sm, Et, precip)")
    ap.add_argument("--outdir",   required=True, help="Output directory")
    ap.add_argument("--i0", type=int, default=0, help="lat start index (inclusive)")
    ap.add_argument("--i1", type=int, default=50, help="lat end index (exclusive)")
    ap.add_argument("--j0", type=int, default=0, help="lon start index (inclusive)")
    ap.add_argument("--j1", type=int, default=50, help="lon end index (exclusive)")
    ap.add_argument("--eps_sign", type=float, default=0.0, help="tolerance for sign neutral (|delta| <= eps -> 0)")
    args = ap.parse_args()

    run_sensitivity(args.dataset, args.variable, args.outdir,
                    i0=args.i0, i1=args.i1, j0=args.j0, j1=args.j1,
                    eps_sign=args.eps_sign)


if __name__ == "__main__":
    main()

    """ python3 01c-sensitivity.py \
            --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr \
            --variable precip \
            --outdir /mnt/data/romi/output/paper_1/output_precip_final \
            --i0 270 --i1 320 --j0 470 --j1 520 """