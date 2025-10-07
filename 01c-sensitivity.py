import os
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.seasonal import STL
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score


""" 
Run a simple sensitivity analysis using cohen's kappa that tests effect of temporal aggregation, 
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
            --i0 270 --i1 320 --j0 470 --j1 520: indices to slice out the tile 
            
            
"""

# ========= helpers to open and prep =========

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
        raise ValueError("Dataset must have lon and lat dims.")
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds

def _infer_agg(variable):
    v = str(variable).lower()
    return "sum" if any(a in v for a in {"precip","pr","precipitation","tp","rain"}) else "mean"

def resample_to(ds, variable, freq):
    """None or 'D' keeps original; 'W' weekly; 'MS' monthly start."""
    if (freq is None) or (str(freq).upper() == "D"):
        return ds
    how = _infer_agg(variable)
    if str(freq).upper().startswith("W"):
        return (ds.resample(time=freq).sum(keep_attrs=True, skipna=True)
                if how == "sum" else
                ds.resample(time=freq).mean(keep_attrs=True, skipna=True))
    if str(freq).upper() in ("M","MS"):
        return (ds.resample(time="MS").sum(keep_attrs=True, skipna=True)
                if how == "sum" else
                ds.resample(time="MS").mean(keep_attrs=True, skipna=True))
    raise ValueError("Unsupported freq. Use D, W, or MS.")

def steps_per_year_from_times(times):
    t = pd.to_datetime(times)
    span_years = (t[-1] - t[0]).days / 365.2425
    return max(3, int(round(len(t) / span_years)))

def stl_period_for(times):
    return steps_per_year_from_times(times)

def make_odd(n):
    return n if (n % 2 == 1) else (n + 1)

# ========= STL detrend =========

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

    seasonal = max(7, min(int(round(period/4)), 61))
    trend    = max(2*period-1, 101)
    res = STL(filled, period=period, seasonal=seasonal, trend=trend, robust=robust).fit().resid
    res[ts.isna()] = np.nan
    return res.values

# ========= rolling EWS =========

def rolling_ews_centered(ts, window):
    n = len(ts); half = window // 2
    out = [np.full(n, np.nan) for _ in range(5)]  # ac1, sd, skew, kurt, fd
    for k in range(half, n - half):
        w = ts[k-half:k+half+1]
        if np.sum(~np.isnan(w)) > 2:
            m  = np.nanmean(w)
            v  = np.nanvar(w, ddof=1)
            s  = np.nanstd(w, ddof=1)
            out[1][k] = s
            out[0][k] = np.nansum((w[1:]-m)*(w[:-1]-m)) / (v*(len(w)-1)) if v > 0 else np.nan
            if s > 0:
                z = (w-m)/s
                out[2][k] = np.nanmean(z**3)
                out[3][k] = np.nanmean(z**4)
            d1 = np.nansum(np.abs(w[1:]-w[:-1])) / (2*(len(w)-1))
            d2 = np.nansum(np.abs(w[2:]-w[:-2])) / (2*(len(w)-2)) if len(w) > 2 else np.nan
            if (d1 is not np.nan) and (d2 is not np.nan) and (d1 > 0) and (d2 > 0):
                x   = np.log([1,2]); xb = x.mean()
                fdv = 2 - ((x[0]-xb)*np.log(d1) + (x[1]-xb)*np.log(d2)) / ((x-xb)**2).sum()
                out[4][k] = np.clip(fdv, 1, 2)
    return tuple(out)

# ========= Kendall tau =========

def kendall_tau_map(arr3):
    """
    arr3: (time, lat, lon) time series of an indicator.
    Returns (tau_map, pval_map) shaped (lat, lon).
    """
    nt, nlat, nlon = arr3.shape
    tau  = np.full((nlat, nlon), np.nan, dtype=np.float32)
    pval = np.full((nlat, nlon), np.nan, dtype=np.float32)
    x = np.arange(nt, dtype=float)
    for i in range(nlat):
        for j in range(nlon):
            y = arr3[:, i, j]
            if np.all(np.isnan(y)) or np.count_nonzero(~np.isnan(y)) < 4:
                continue
            t, p = kendalltau(x, y, nan_policy="omit")
            tau[i, j]  = t
            pval[i, j] = p
    return tau, pval

def label_from_tau(tau_map, eps_tau):
    """
    Map tau to {-1, 0, +1} with absolute value <= eps considered neutral.
    Missing tau remains missing and does not count as neutral.
    """
    tau = np.asarray(tau_map)
    lbl = np.full(tau.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(tau)
    lbl[finite & (tau >  eps_tau)] =  1.0
    lbl[finite & (tau < -eps_tau)] = -1.0
    lbl[finite & (np.abs(tau) <= eps_tau)] = 0.0
    return lbl

# ========= core compute for a tile =========

def compute_ews_for_tile(raw3, times, freq, years_window,
                         do_diff=True, stl_robust=True, fill="mean"):
    nt, nlat, nlon = raw3.shape
    period = stl_period_for(times)

    flat = raw3.reshape(nt, -1)
    resid = np.empty_like(flat)
    for i in range(flat.shape[1]):
        resid[:, i] = stl_residual(flat[:, i], times, period=period, robust=stl_robust, fill=fill)
    resid3 = resid.reshape(nt, nlat, nlon)

    if do_diff:
        detr = np.empty_like(resid3)
        detr[0] = np.nan
        detr[1:] = resid3[1:] - resid3[:-1]
    else:
        detr = resid3.copy()

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

    data_vars = {
        "raw":       (("time","lat","lon"), raw3),
        "residual":  (("time","lat","lon"), resid3),
        "detrended": (("time","lat","lon"), detr),
        "ac1":       (("time","lat","lon"), ac1),
        "std":       (("time","lat","lon"), sd),
        "skew":      (("time","lat","lon"), skew),
        "kurt":      (("time","lat","lon"), kurt),
        "fd":        (("time","lat","lon"), fd),
    }
    coords = {"time": times, "lat": np.arange(nlat), "lon": np.arange(nlon)}
    return xr.Dataset(data_vars, coords=coords)

# ========= tile I/O =========

def load_tile(dataset_path, variable, freq_resample, i0, i1, j0, j1,
              t0="2000-01-01", t1="2023-12-31"):
    ds = open_source_ds(dataset_path)
    ds = standardize_lat_lon(ds)
    if "time" not in ds:
        raise ValueError("Dataset has no time coordinate.")
    ds = ds.sel(time=slice(np.datetime64(t0), np.datetime64(t1)))

    ds_r = resample_to(ds, variable, freq_resample)

    da = ds_r[variable].isel(lat=slice(i0, i1), lon=slice(j0, j1)).transpose("time","lat","lon")
    raw3  = da.values
    times = pd.to_datetime(da["time"].values)
    ds.close()
    return raw3, times

def safe_name(cfg):
    tag = f"freq{cfg['freq']}_win{cfg['win_years']}y_diff{int(cfg['diff'])}_stl{'R' if cfg['stl_robust'] else 'N'}"
    return tag.replace(".","p")

# ========= Cohen's kappa (pairwise) and Light's kappa =========

def _pair_kappa_unweighted(a, b):
    """
    a, b: 1D arrays with values in {-1, 0, 1, NaN}
    Returns unweighted Cohen's kappa.
    Handles single-class cases and always passes the full label set.
    """
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan

    A = a[m].astype(int)
    B = b[m].astype(int)

    # if both raters are constant (only one label each)
    ua = np.unique(A)
    ub = np.unique(B)
    if ua.size == 1 and ub.size == 1:
        # same constant label → perfect agreement
        if ua[0] == ub[0]:
            return 1.0
        # different constant labels → no agreement
        return 0.0

    # general case; pass all labels to avoid warnings
    return cohen_kappa_score(A, B, labels=[-1, 0, 1])


def kappa_matrix(label_maps):
    """
    label_maps: {config_tag: 2D label array with values in {-1, 0, 1, NaN}}
    Returns a symmetric DataFrame of unweighted Cohen's kappa over all pairs.
    """
    tags = list(label_maps.keys())
    n = len(tags)
    mat = np.full((n, n), np.nan, dtype=float)
    for i, ti in enumerate(tags):
        li = label_maps[ti].ravel()
        for j, tj in enumerate(tags):
            if j < i:
                mat[i, j] = mat[j, i]
                continue
            lj = label_maps[tj].ravel()
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = _pair_kappa_unweighted(li, lj)
    return pd.DataFrame(mat, index=tags, columns=tags)

def lights_kappa_from_matrix(kdf):
    """Mean of the upper triangle excluding diagonal."""
    vals = kdf.where(~np.eye(len(kdf), dtype=bool)).to_numpy()
    uptri = vals[np.triu_indices_from(vals, k=1)]
    return float(np.nanmean(uptri)) if np.isfinite(uptri).any() else np.nan

# ========= sensitivity driver =========

def run_sensitivity(dataset, variable, outdir,
                    i0=0, i1=50, j0=0, j1=50,
                    freqs=("W","MS"),
                    windows_years=(2.5, 5.0, 10.0),
                    diffs=(True, False),
                    stl_robust_opts=(True, False),
                    fill="mean",
                    eps_tau=0.05):
    os.makedirs(outdir, exist_ok=True)

    SENSITIVITIES = []
    for f in freqs:
        for wy in windows_years:
            for d in diffs:
                for r in stl_robust_opts:
                    SENSITIVITIES.append({"freq": f, "win_years": wy, "diff": d, "stl_robust": r})

    cache = {}
    for f in freqs:
        raw3, times = load_tile(dataset, variable, f, i0, i1, j0, j1)
        cache[f] = (raw3, times)

    by_indicator_labels = { "ac1": {}, "std": {}, "skew": {}, "kurt": {}, "fd": {} }

    tau_summaries = []

    for cfg in SENSITIVITIES:
        f, wy, dff, rob = cfg["freq"], cfg["win_years"], cfg["diff"], cfg["stl_robust"]
        raw3, times = cache[f]
        ds_out = compute_ews_for_tile(raw3, times, f, wy, do_diff=dff, stl_robust=rob, fill=fill)

        tau_maps, pval_maps = {}, {}
        for ind in ("ac1","std","skew","kurt","fd"):
            tau, p = kendall_tau_map(ds_out[ind].values)
            tau_maps[ind], pval_maps[ind] = tau, p

            tau_summaries.append({
                "freq": f, "win_years": wy, "diff": dff, "stl_robust": rob,
                "indicator": ind,
                "tau_mean": float(np.nanmean(tau)),
                "tau_median": float(np.nanmedian(tau)),
                "tau_std": float(np.nanstd(tau)),
                "pct_pos": float(np.nanmean(tau >  eps_tau) * 100.0),
                "pct_neg": float(np.nanmean(tau < -eps_tau) * 100.0),
                "pct_neu": float(np.nanmean(np.abs(tau) <= eps_tau) * 100.0),
            })

        tag = safe_name(cfg)
        out_path = os.path.join(outdir, f"sensitivity_{variable}_{tag}_kt.zarr")
        if os.path.exists(out_path):
            import shutil; shutil.rmtree(out_path)

        kt_ds = xr.Dataset(
            data_vars={
                "tau_ac1":  (("lat","lon"), tau_maps["ac1"]),
                "tau_std":  (("lat","lon"), tau_maps["std"]),
                "tau_skew": (("lat","lon"), tau_maps["skew"]),
                "tau_kurt": (("lat","lon"), tau_maps["kurt"]),
                "tau_fd":   (("lat","lon"), tau_maps["fd"]),
                "p_ac1":    (("lat","lon"), pval_maps["ac1"]),
                "p_std":    (("lat","lon"), pval_maps["std"]),
                "p_skew":   (("lat","lon"), pval_maps["skew"]),
                "p_kurt":   (("lat","lon"), pval_maps["kurt"]),
                "p_fd":     (("lat","lon"), pval_maps["fd"]),
            },
            coords={"lat": ds_out["lat"].values, "lon": ds_out["lon"].values}
        )
        kt_ds.to_zarr(out_path, mode="w", consolidated=True)
        print("[OK] wrote", out_path)

        for ind in by_indicator_labels.keys():
            by_indicator_labels[ind][tag] = label_from_tau(tau_maps[ind], eps_tau)

    df_tau = pd.DataFrame(tau_summaries)
    df_tau_path = os.path.join(outdir, f"{variable}_kt_summary.csv")
    df_tau.to_csv(df_tau_path, index=False)
    print("[DONE] tau summary saved to:", df_tau_path)

    lk_rows = []
    for ind, lbls in by_indicator_labels.items():
        kdf = kappa_matrix(lbls)
        kappa_csv = os.path.join(outdir, f"{variable}_cohens_kappa_{ind}.csv")
        kdf.to_csv(kappa_csv)
        print("[DONE] kappa matrix", ind, "saved to:", kappa_csv)

        lk = lights_kappa_from_matrix(kdf)
        lk_rows.append({"indicator": ind, "lights_kappa": lk, "n_configs": len(lbls), "n_pairs": int(len(lbls)*(len(lbls)-1)/2)})

    lk_df = pd.DataFrame(lk_rows).sort_values("lights_kappa", ascending=False)
    lk_df["robustness_rank"] = np.arange(1, len(lk_df)+1)
    lk_csv = os.path.join(outdir, f"{variable}_lights_kappa_summary.csv")
    lk_df.to_csv(lk_csv, index=False)
    print("[DONE] Light's kappa summary saved to:", lk_csv)

# ========= CLI =========

def main():
    ap = argparse.ArgumentParser(description="Sensitivity via Kendall tau labels and Light's kappa including neutrals.")
    ap.add_argument("--dataset",  required=True, help="Path to full dataset (zarr or netcdf)")
    ap.add_argument("--variable", required=True, help="Variable name (e.g., sm, Et, precip)")
    ap.add_argument("--outdir",   required=True, help="Output directory")
    ap.add_argument("--i0", type=int, default=0, help="lat start index (inclusive)")
    ap.add_argument("--i1", type=int, default=50, help="lat end index (exclusive)")
    ap.add_argument("--j0", type=int, default=0, help="lon start index (inclusive)")
    ap.add_argument("--j1", type=int, default=50, help="lon end index (exclusive)")
    ap.add_argument("--eps_tau", type=float, default=0.05, help="neutral threshold for tau (absolute value <= eps is 0)")
    args = ap.parse_args()

    run_sensitivity(args.dataset, args.variable, args.outdir,
                    i0=args.i0, i1=args.i1, j0=args.j0, j1=args.j1,
                    eps_tau=args.eps_tau)

if __name__ == "__main__":
    main()

"""
Example:
python3 01c-sensitivity_tau_lightsk.py \
  --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr \
  --variable precip \
  --outdir /mnt/data/romi/output/paper_1/output_precip_final \
  --i0 270 --i1 320 --j0 470 --j1 520 \
  --eps_tau 0.05
"""
