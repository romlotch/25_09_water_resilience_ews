#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score
import re
from utils.config import load_config, cfg_path


"""
Example:
python3 01c-sensitivity.py \
  --dataset /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr \
  --variable precip \
  --outdir /mnt/data/romi/output/paper_1/output_precip_final \
  --i0 270 --i1 320 --j0 470 --j1 520 \
  --eps_tau 0.05 --p_alpha 0.05
"""


# ----- open and prep helpers -----

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

# ----- STL -----

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

# ----- Rolling EWS -----

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

# ----- Kendall tau -----

def kendall_tau_map(arr3):
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

def label_from_tau_and_p(tau_map, p_map, eps_tau, p_alpha):
    """
    Rules:
      - if p > p_alpha -> neutral (0)
      - else use sign of tau with epsilon threshold
      - missing p or tau -> NaN (excluded from pairwise comparison)
    """
    tau = np.asarray(tau_map)
    p   = np.asarray(p_map)
    lbl = np.full(tau.shape, np.nan, dtype=np.float32)

    finite_both = np.isfinite(tau) & np.isfinite(p)

    # neutral by non-significance
    neutral_mask = finite_both & (p > p_alpha)
    lbl[neutral_mask] = 0.0

    # significant: classify by tau magnitude and sign
    sig_mask = finite_both & (p <= p_alpha)
    lbl[sig_mask & (tau >  eps_tau)] =  1.0
    lbl[sig_mask & (tau < -eps_tau)] = -1.0
    lbl[sig_mask & (np.abs(tau) <= eps_tau)] = 0.0

    return lbl

# ----- compute EWS for a tile -----

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

# ----- Cohen's kappa and Light's kappa -----

def _pair_kappa_unweighted(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    A = a[m].astype(int)
    B = b[m].astype(int)
    ua = np.unique(A)
    ub = np.unique(B)
    if ua.size == 1 and ub.size == 1:
        if ua[0] == ub[0]:
            return 1.0
        return 0.0
    return cohen_kappa_score(A, B, labels=[-1, 0, 1])

def kappa_matrix(label_maps):
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
    vals = kdf.where(~np.eye(len(kdf), dtype=bool)).to_numpy()
    uptri = vals[np.triu_indices_from(vals, k=1)]
    return float(np.nanmean(uptri)) if np.isfinite(uptri).any() else np.nan

def kappa_distribution(kdf):
    arr = kdf.to_numpy()
    n = arr.shape[0]
    up = arr[np.triu_indices(n, k=1)]
    up = up[np.isfinite(up)]
    if up.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.percentile(up, 10)), float(np.median(up)), float(np.percentile(up, 90))

def best_worst_pairs(kdf, top=3):
    recs = []
    idx = list(kdf.index)
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            recs.append((idx[i], idx[j], kdf.iat[i, j]))
    df = pd.DataFrame(recs, columns=["a","b","kappa"]).dropna()
    if df.empty:
        return pd.DataFrame(columns=["a","b","kappa"]), pd.DataFrame(columns=["a","b","kappa"])
    df_sorted = df.sort_values("kappa", ascending=False)
    return df_sorted.head(top), df_sorted.tail(top)

_tag_pat = re.compile(r"freq(?P<freq>W|MS)_win(?P<win>[\dp]+)y_diff(?P<diff>[01])_stl(?P<stl>[RN])")

def parse_tag(tag):
    m = _tag_pat.fullmatch(tag)
    if not m:
        return {"freq": None, "win": None, "diff": None, "stl": None}
    d = m.groupdict()
    d["win"] = d["win"].replace("p", ".")
    return d



def mean_when_only_one_factor_changes(kdf):
    names = list(kdf.index)
    meta = pd.DataFrame([parse_tag(n) for n in names], index=names)
    recs = {"freq": [], "win": [], "diff": [], "stl": []}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            va, vb = meta.loc[a], meta.loc[b]
            eq = [(va["freq"] == vb["freq"]),
                  (va["win"]  == vb["win"]),
                  (va["diff"] == vb["diff"]),
                  (va["stl"]  == vb["stl"])]
            if sum(eq) == 3:
                if va["freq"] != vb["freq"]:
                    recs["freq"].append(kdf.loc[a, b])
                elif va["win"] != vb["win"]:
                    recs["win"].append(kdf.loc[a, b])
                elif va["diff"] != vb["diff"]:
                    recs["diff"].append(kdf.loc[a, b])
                elif va["stl"] != vb["stl"]:
                    recs["stl"].append(kdf.loc[a, b])
    out = {k: float(np.nanmean(v)) if len(v) else np.nan for k, v in recs.items()}
    return pd.Series(out, name="mean_kappa_when_only_one_factor_changes")



def lights_kappa_by_freq(kdf):
    """Return Light's kappa computed on Weekly-only and Monthly-only config subsets."""
    names = list(kdf.index)
    meta = pd.DataFrame([parse_tag(n) for n in names], index=names)
    out = {}
    for freq in ["W", "MS"]:
        idx = [n for n in names if meta.loc[n, "freq"] == freq]
        if len(idx) >= 2:
            sub = kdf.loc[idx, idx]
            out[f"lights_kappa_{freq}"] = lights_kappa_from_matrix(sub)
        else:
            out[f"lights_kappa_{freq}"] = np.nan
    return pd.Series(out)



def mean_when_only_one_factor_changes_by_freq(kdf):
    """
    Like mean_when_only_one_factor_changes, but split into Weekly-only and Monthly-only.
    Pairs differ in exactly one factor (win, diff, or stl) and share the same freq.
    """
    names = list(kdf.index)
    meta = pd.DataFrame([parse_tag(n) for n in names], index=names)
    buckets = {"W": {"win": [], "diff": [], "stl": []},
               "MS": {"win": [], "diff": [], "stl": []}}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            va, vb = meta.loc[a], meta.loc[b]
            if va["freq"] != vb["freq"]:
                continue  # only within same frequency
            eq = [(va["win"]  == vb["win"]),
                  (va["diff"] == vb["diff"]),
                  (va["stl"]  == vb["stl"])]
            if sum(eq) == 2:  # differ in exactly one
                if va["win"] != vb["win"]:
                    buckets[va["freq"]]["win"].append(kdf.loc[a, b])
                elif va["diff"] != vb["diff"]:
                    buckets[va["freq"]]["diff"].append(kdf.loc[a, b])
                elif va["stl"] != vb["stl"]:
                    buckets[va["freq"]]["stl"].append(kdf.loc[a, b])
    rows = []
    for freq in ["W","MS"]:
        rows.append({
            "freq": freq,
            "mean_kappa_win_changes":  float(np.nanmean(buckets[freq]["win"]))  if buckets[freq]["win"]  else np.nan,
            "mean_kappa_diff_changes": float(np.nanmean(buckets[freq]["diff"])) if buckets[freq]["diff"] else np.nan,
            "mean_kappa_stl_changes":  float(np.nanmean(buckets[freq]["stl"]))  if buckets[freq]["stl"]  else np.nan,
        })
    return pd.DataFrame(rows).set_index("freq")



# ----- Run sensitivity -----

def run_sensitivity(dataset, variable, outdir,
                    i0=0, i1=50, j0=0, j1=50,
                    freqs=("W","MS"),
                    windows_years=(2.5, 5.0, 10.0),
                    diffs=(True, False),
                    stl_robust_opts=(True, False),
                    fill="mean",
                    eps_tau=0.05,
                    p_alpha=0.05,
                    report_top_pairs=3):
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
                "pct_pos_tau_gt_eps": float(np.nanmean(tau >  eps_tau) * 100.0),
                "pct_neg_tau_lt_minus_eps": float(np.nanmean(tau < -eps_tau) * 100.0),
                "pct_neu_abs_tau_le_eps": float(np.nanmean(np.abs(tau) <= eps_tau) * 100.0),
                "pct_p_greater_alpha": float(np.nanmean(p > p_alpha) * 100.0)
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
            lbl = label_from_tau_and_p(tau_maps[ind], pval_maps[ind], eps_tau, p_alpha)
            by_indicator_labels[ind][tag] = lbl

    df_tau = pd.DataFrame(tau_summaries)
    df_tau_path = os.path.join(outdir, f"{variable}_kt_summary.csv")
    df_tau.to_csv(df_tau_path, index=False)
    print("[DONE] tau summary saved to:", df_tau_path)

    # Kappa summaries and ranking per indicator
    detail_rows = []
    lk_rows = []

    for ind, lbls in by_indicator_labels.items():
        kdf = kappa_matrix(lbls)
        kappa_csv = os.path.join(outdir, f"{variable}_cohens_kappa_{ind}.csv")
        kdf.to_csv(kappa_csv)
        print("[DONE] kappa matrix", ind, "saved to:", kappa_csv)

        lk_byfreq = lights_kappa_by_freq(kdf)
        byfreq_means = mean_when_only_one_factor_changes_by_freq(kdf)

        lk = lights_kappa_from_matrix(kdf)
        p10, med, p90 = kappa_distribution(kdf)
        best_df, worst_df = best_worst_pairs(kdf, top=report_top_pairs)
        factor_means = mean_when_only_one_factor_changes(kdf)

        # store detailed row for indicator
        detail_rows.append({
            "indicator": ind,
            "lights_kappa": lk,
            "kappa_p10": p10,
            "kappa_median": med,
            "kappa_p90": p90,
            "mean_kappa_when_only_freq_changes": factor_means.get("freq", np.nan),
            "mean_kappa_when_only_win_changes":  factor_means.get("win",  np.nan),
            "mean_kappa_when_only_diff_changes": factor_means.get("diff", np.nan),
            "mean_kappa_when_only_stl_changes":  factor_means.get("stl",  np.nan),
            "W_mean_kappa_win_changes":  byfreq_means.loc["W","mean_kappa_win_changes"],
            "W_mean_kappa_diff_changes": byfreq_means.loc["W","mean_kappa_diff_changes"],
            "W_mean_kappa_stl_changes":  byfreq_means.loc["W","mean_kappa_stl_changes"],
            "MS_mean_kappa_win_changes":  byfreq_means.loc["MS","mean_kappa_win_changes"],
            "MS_mean_kappa_diff_changes": byfreq_means.loc["MS","mean_kappa_diff_changes"],
            "MS_mean_kappa_stl_changes":  byfreq_means.loc["MS","mean_kappa_stl_changes"],
            "best_pair_1": best_df.iloc[0]["a"] if len(best_df) > 0 else "",
            "best_pair_2": best_df.iloc[0]["b"] if len(best_df) > 0 else "",
            "best_pair_kappa": float(best_df.iloc[0]["kappa"]) if len(best_df) > 0 else np.nan,
            "worst_pair_1": worst_df.iloc[0]["a"] if len(worst_df) > 0 else "",
            "worst_pair_2": worst_df.iloc[0]["b"] if len(worst_df) > 0 else "",
            "worst_pair_kappa": float(worst_df.iloc[0]["kappa"]) if len(worst_df) > 0 else np.nan,
            "n_configs": len(lbls),
            "n_pairs": int(len(lbls) * (len(lbls) - 1) / 2)
        })

        lk_rows.append({"indicator": ind, "lights_kappa": lk, "n_configs": len(lbls), "n_pairs": int(len(lbls)*(len(lbls)-1)/2)})

    detail_df = pd.DataFrame(detail_rows).sort_values("lights_kappa", ascending=False)
    detail_csv = os.path.join(outdir, f"{variable}_indicator_kappa_detail.csv")
    detail_df.to_csv(detail_csv, index=False)
    print("[DONE] indicator detail summary saved to:", detail_csv)

    lk_df = pd.DataFrame(lk_rows).sort_values("lights_kappa", ascending=False)
    lk_df["robustness_rank"] = np.arange(1, len(lk_df)+1)
    lk_csv = os.path.join(outdir, f"{variable}_lights_kappa_summary.csv")
    lk_df.to_csv(lk_csv, index=False)
    print("[DONE] Light's kappa summary saved to:", lk_csv)

# CLI

def main():
    p = argparse.ArgumentParser(description="Sensitivity via Kendall tau labels and Light's kappa including neutrals, plus summaries.")

    p.add_argument("--variable", required=True, help="Variable name (e.g., sm, Et, precip)")
    p.add_argument("--dataset", default=None,
               help="Optional override: path to full dataset (zarr or netcdf). If omitted, inferred from config + --variable + --suffix.")
    p.add_argument("--suffix", default=None,
                help="Optional filename suffix (e.g. 'breakpoint_stc' -> out_<var>_breakpoint_stc.zarr) used when inferring --dataset.")
    p.add_argument("--outdir", default=None,
                help="Optional override output directory. If omitted, uses outputs/tables/sensitivity/<var>/ under outputs_root.")
    p.add_argument("--i0", type=int, default=0, help="lat start index (inclusive)")
    p.add_argument("--i1", type=int, default=50, help="lat end index (exclusive)")
    p.add_argument("--j0", type=int, default=0, help="lon start index (inclusive)")
    p.add_argument("--j1", type=int, default=50, help="lon end index (exclusive)")
    p.add_argument("--eps_tau", type=float, default=0.05, help="neutral threshold for tau magnitude")
    p.add_argument("--p_alpha", type=float, default=0.05, help="p value threshold; p > alpha becomes neutral")
    p.add_argument("--report_top_pairs", type=int, default=3, help="how many best/worst pairs to record per indicator")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)

    # default dataset path: outputs/zarr/out_<var><_suffix>.zarr
    sfx = "" if not args.suffix else (args.suffix if args.suffix.startswith("_") else f"_{args.suffix}")
    default_ds = Path(outputs_root) / "zarr" / f"out_{args.variable}{sfx}.zarr"
    dataset = args.dataset or str(default_ds)

    # default outdir: outputs/tables/sensitivity/<var>/
    default_outdir = Path(outputs_root) / "tables" / "sensitivity" / args.variable
    outdir = args.outdir or str(default_outdir)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    run_sensitivity(dataset, args.variable, outdir,
                i0=args.i0, i1=args.i1, j0=args.j0, j1=args.j1,
                eps_tau=args.eps_tau, p_alpha=args.p_alpha,
                report_top_pairs=args.report_top_pairs)

if __name__ == "__main__":
    main()
