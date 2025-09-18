import os
import time
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

import geopandas as gpd
from shapely.geometry import Point

import matplotlib
import matplotlib as mpl
from matplotlib import colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sn

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import shap

"""
Predict occurance of abrupt shifts in variable time series with 
kendall tau of EWS and environmental variables. Input target tau, pre-breakpoint tau, 
and changepoints datasets. Filepaths or variables that are hardcoded are in pold.

Runs H2O XGBoost with 5-fold CV. 

Inputs: 
    - tau_full: path to the full kendall tau output for the variable 
    - tau_pre: path to the kendall tau output before the breakpoint 
    - breaks: path to the output of the breakpoints anslysis for the abrupt change test
    - var: variable name (e.g. sm, Et, precip)
    - cp_test: specify 'pettitt', 'stc', or 'var'. Should match the breakpoint and tau-pre path. 
    - outdir: directory to save the results and figures to
                    
Plots saved as SVGs: 
    - correlation heatmap
    - calibration curve
    - precision-recall 
    - feature importances
    - PDPs of top-5 features

E.g. 

    python 06b-run_random_forest_scipy.py  --tau_full '/home/romi/ews/output/output_sm_final/out_sm_kt.zarr' \
                                     --tau_pre  '/home/romi/ews/output/output_sm_final/out_sm_breakpoint_stc_kt.zarr' \
                                     --breaks   '/home/romi/ews/output/output_sm_final/out_sm_chp.zarr' \
                                     --tau_pre_neg '/home/romi/ews/output/output_sm_final/out_sm_breakpoint_stc_neg_kt.zarr' \
                                     --var      'sm' \
                                     --cp_test  'stc' \
                                     --outdir   '/home/romi/ews/figures/xgboost_results'

    python 06b-run_random_forest_scipy.py  --tau_full '/home/romi/ews/output/output_Et_final/out_Et_kt.zarr' \
                                     --tau_pre  '/home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc_kt.zarr' \
                                     --breaks   '/home/romi/ews/output/output_Et_final/out_Et_chp.zarr' \
                                     --tau_pre_neg '/home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc_neg_kt.zarr' \
                                     --var      'Et' \
                                     --cp_test  'stc' \
                                     --outdir   '/home/romi/ews/figures/xgboost_results'

    python 06b-run_random_forest_scipy.py  --tau_full '/home/romi/ews/output/output_precip_final/out_precip_kt.zarr' \
                                     --tau_pre  '/home/romi/ews/output/output_precip_final/out_precip_breakpoint_stc_kt.zarr' \
                                     --tau_pre_neg '/home/romi/ews/output/output_precip_final/out_precip_breakpoint_stc_neg_kt.zarr' \
                                     --breaks   '/home/romi/ews/output/output_precip_final/out_precip_chp.zarr' \
                                     --var      'precip' \
                                     --cp_test  'stc' \
                                     --outdir   '/home/romi/ews/figures/xgboost_results'
                                 

"""


# ------------------------------
# Config things
# ------------------------------

CP_PVAL_MAP = {
    "pettitt": {"cp": "pettitt_cp",     "pval": "pettitt_pval", "label": "Pettitt"},
    "stc":     {"cp": "strucchange_bp", "pval": "Fstat_pval",   "label": "StructChange F"},
    "var":     {"cp": "bp_var",         "pval": "pval_var",     "label": "Variance break"},
}

ARIDITY_BINS   = [0, 0.05, 0.20, 0.50, 0.65, np.inf]
ARIDITY_LABELS = ["Hyper-arid", "Arid", "Semi-arid", "Dry subhumid", "Humid"]

QUICK = True
RANDOM_STATE = 42
# ------------------------------
# Helpers
# ------------------------------

def ensure_lat_lon(ds):
    ren = {}
    if "latitude" in ds.dims: ren["latitude"] = "lat"
    if "longitude" in ds.dims: ren["longitude"] = "lon"
    if "y" in ds.dims: ren["y"] = "lat"
    if "x" in ds.dims: ren["x"] = "lon"
    return ds.rename(ren) if ren else ds

def _standardize_ll(ds):
    ds = ensure_lat_lon(ds)
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds

def detect_lon_lat_names(ds):
    lon = next((c for c in ["lon","longitude","x"] if c in ds.coords), None)
    lat = next((c for c in ["lat","latitude","y"] if c in ds.coords), None)
    if lon is None or lat is None:
        raise ValueError(f"Could not find lon/lat in coords: {list(ds.coords)}")
    return lon, lat


def compute_aridity_classes_like(ds_target):
    """
    Compute time-mean aridity index (precip/PET) and bin to ARIDITY_LABELS on ds_target grid.
    Uses ERA5 precip (mm/month) and a PET monthly dataset.
    """
    AI_MEAN_MAX = 100.0        

    def wrap_to_180(ds, lon_name="lon"):
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
        return ds

    # --- Precip (ERA5 monthly total precip 2000–2023) ---
    precip = xr.open_dataset(
        "/home/romi/ews/data/total_precipitation_monthly.nc"
    ).sel(time=slice("2000-01-01", "2023-12-31"))
    precip = precip.rename({"latitude": "lat", "longitude": "lon"})
    precip = precip.rio.write_crs("EPSG:4326")
    precip = wrap_to_180(precip, "lon")
    precip.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Land-sea mask to drop ocean
    ds_mask = xr.open_dataset("/home/romi/ews/data/landsea_mask.grib")
    ds_mask = ds_mask.rename({"latitude": "lat", "longitude": "lon"})
    ds_mask = ds_mask.rio.write_crs("EPSG:4326")
    ds_mask = wrap_to_180(ds_mask, "lon")
    ds_mask.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Interp mask to precip grid and apply
    mask_interp = ds_mask["lsm"].interp(lat=precip.lat, lon=precip.lon)
    precip = precip.where(mask_interp > 0.7)

    # m/month → mm/month
    precip = precip * 1000.0
    precip = precip.drop_vars(["number", "step", "surface", "spatial_ref"], errors="ignore")

    # Force time to month-start
    precip["time"] = ("time", pd.date_range(
        start=str(precip.time.values[0])[:10],
        periods=precip.sizes["time"],
        freq="MS"
    ))

    # --- PET (monthly) ---
    pet = xr.open_dataset("/home/romi/ews/data/monthly_sum_epot_clean.zarr").sel(
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
    ai_mean = ai_annual.mean("time")

    lon, lat = detect_lon_lat_names(ds_target)
    ai_on_target = ai_mean.interp(lat=ds_target[lat], lon=ds_target[lon])

    idx = xr.apply_ufunc(np.digitize, ai_on_target, kwargs={"bins": ARIDITY_BINS},
                         vectorize=True, dask="parallelized", output_dtypes=[int])

    def idx2label(i):
        return ARIDITY_LABELS[i-1] if (1 <= i <= len(ARIDITY_LABELS)) else np.nan

    labels = xr.apply_ufunc(np.vectorize(idx2label), idx,
                            vectorize=True, dask="parallelized", output_dtypes=[object]).rename("aridity_class")

    # Ensure lat/lon names match
    if lon != "lon" or lat != "lat":
        labels = labels.rename({lon:"lon", lat:"lat"})
    return labels


def masked_zarr_path(tau_full_path, var, cp_test):
    p = Path(tau_full_path)
    return str(p.parent / f"{var}_cp_masked_{cp_test.lower()}.zarr")

def ews_base_path_from_tau_full(tau_full_path):
    p = Path(tau_full_path)
    name = p.name.replace("_kt", "")
    return str(p.with_name(name))

def load_and_harmonize_predictor(path, target_lat, target_lon, vars_to_drop):
    ds_pred = xr.open_dataset(path)
    ds_pred = ensure_lat_lon(ds_pred)
    if "datetime" in ds_pred.dims:
        ds_pred = ds_pred.rename({"datetime": "time"})
    if ("lat" in ds_pred.coords) and ("lon" in ds_pred.coords):
        ds_pred = ds_pred.assign_coords(lat=("lat", ds_pred["lat"].values),
                                        lon=("lon", ds_pred["lon"].values))
        ds_pred = ds_pred.interp(lat=target_lat, lon=target_lon)

    for dim in list(ds_pred.dims):
        if dim not in ("lat", "lon"):
            ds_pred = ds_pred.isel({dim: 0}, drop=True) if ds_pred.sizes[dim] == 1 else ds_pred.mean(dim)

    extra_coords = [c for c in ds_pred.coords if c not in ("lat", "lon")]
    if extra_coords:
        ds_pred = ds_pred.drop_vars(extra_coords)

    basename = os.path.basename(path).replace(".zarr", "")
    prefix = basename.replace("driver_", "")
    ds_pred = ds_pred.rename({v: f"{prefix}_{v}" for v in ds_pred.data_vars})

    for var in vars_to_drop:
        if var in ds_pred.data_vars: ds_pred = ds_pred.drop_vars(var)
        if var in ds_pred.coords:    ds_pred = ds_pred.drop_vars(var)
    return ds_pred

def align_to_train(hf_any, train_types, train_cols):
    # Add missing columns as NA
    for c in train_cols:
        if c not in hf_any.columns:
            hf_any[c] = None
    # Drop extras and reorder
    hf_any = hf_any[train_cols]
    # Match dtypes
    for c, t in train_types.items():
        if t == "enum":
            hf_any[c] = hf_any[c].asfactor()
        else:
            hf_any[c] = hf_any[c].asnumeric()
    return hf_any


def build_target_dataframe(tau_full_path, tau_pre_pos_path, breaks_path, var_prefix, cp_key, alpha=0.05, tau_pre_neg_path=None):
    cp_key = cp_key.lower()
    if cp_key not in CP_PVAL_MAP:
        raise ValueError(f"--cp-test must be one of {list(CP_PVAL_MAP.keys())}")

    ds_tau_full    = _standardize_ll(xr.open_dataset(tau_full_path))
    ds_tau_pre_pos = _standardize_ll(xr.open_dataset(tau_pre_pos_path)).interp(lat=ds_tau_full["lat"], lon=ds_tau_full["lon"])

    ds_tau_pre_neg = None
    if tau_pre_neg_path is not None and os.path.exists(tau_pre_neg_path):
        ds_tau_pre_neg = _standardize_ll(xr.open_dataset(tau_pre_neg_path)).interp(lat=ds_tau_full["lat"], lon=ds_tau_full["lon"])

    masked_path = masked_zarr_path(tau_full_path, var_prefix, cp_key)
    ds_masked   = _standardize_ll(xr.open_dataset(masked_path)).interp(lat=ds_tau_full["lat"], lon=ds_tau_full["lon"])
    mask_pos    = ds_masked[var_prefix].notnull().any(dim="time")

    base_path = ews_base_path_from_tau_full(tau_full_path)
    ds_base   = xr.open_dataset(base_path)
    have_raw  = ds_base[f"{var_prefix}"].notnull().any(dim="time")

    ds_lm = xr.open_dataset("/home/romi/ews/data/landsea_mask.grib", engine="cfgrib")
    ds_lm = ensure_lat_lon(ds_lm)
    ds_lm = ds_lm.rio.write_crs('EPSG:4326')
    ds_lm = ds_lm.assign_coords(lon=(((ds_lm.lon + 180) % 360) - 180)).sortby('lon')
    ds_lm.rio.set_spatial_dims("lon", "lat", inplace=True)
    ds_lm = ds_lm.interp(lat=ds_tau_full["lat"], lon=ds_tau_full["lon"])
    lm_name = list(ds_lm.data_vars)[0]
    land_da = (ds_lm[lm_name] > 0.7)

    domain_da = have_raw & land_da
    mask_neg = (~mask_pos) & domain_da

    ds_tau = xr.Dataset(coords=ds_tau_full.coords)
    for v in ds_tau_full.data_vars:
        if not v.startswith(f"{var_prefix}_"):
            continue
        t_full = ds_tau_full[v]
        t_pos  = ds_tau_pre_pos[v] if v in ds_tau_pre_pos.data_vars else None
        t_neg  = (ds_tau_pre_neg[v] if (ds_tau_pre_neg is not None and v in ds_tau_pre_neg.data_vars) else None)

        pos_ok = (t_pos is not None) and (~t_pos.isnull())
        neg_ok = (t_neg is not None) and (~t_neg.isnull())

        choice = xr.where(mask_pos & pos_ok, (t_pos if t_pos is not None else t_full), t_full)
        if t_neg is not None:
            choice = xr.where(mask_neg & neg_ok, t_neg, choice)
        ds_tau[v] = choice

    df_tau = ds_tau.to_dataframe().reset_index()
    target_da = xr.where(domain_da, mask_pos, np.nan)
    df_target = target_da.to_dataset(name="target").to_dataframe().reset_index()

    df_all = df_tau.merge(df_target, on=["lat", "lon"], how="left")
    df_all = df_all[df_all["target"].notna()].copy()
    df_all["target"] = df_all["target"].astype(bool)

    ews_stats = ["ac1", "std", "skew", "kurt", "fd"]
    pval_cols = [f"{var_prefix}_{stat}_pval" for stat in ews_stats if f"{var_prefix}_{stat}_pval" in df_all.columns]
    df_all = df_all.drop(columns=pval_cols, errors="ignore")
    return df_all, ds_tau


def make_balanced_sample(df_all, pos_n, neg_n, seed = 42):

    df_pos = df_all[df_all["target"]].sample(min(pos_n, (df_all["target"] == True).sum()), random_state=seed)
    df_neg = df_all[~df_all["target"]].sample(min(neg_n, (df_all["target"] == False).sum()), random_state=seed)
    df_model = pd.concat([df_pos, df_neg], ignore_index=True)

    return df_model


# ------------------------------
# Diagnostic plots 
# ------------------------------


def plot_and_save_corr_heatmap(df, features, out_svg):
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sn.heatmap(corr_matrix, cmap="RdBu_r", center=0)
    plt.title("Feature Cross-Correlation")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg"); plt.close()

def plot_and_save_calibration(y_true, y_prob, out_svg):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve"); plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(out_svg, format="svg"); plt.close()

def plot_and_save_varimp(varimp_df, out_svg, top_n=10):
    top = varimp_df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    sn.barplot(x="importance", y="variable", data=top)
    plt.xlabel("Importance"); plt.ylabel("Feature"); plt.title(f"Top {top_n} Feature Importance")
    ax = plt.gca()
    for side in ("top","right","bottom","left"): ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout(); plt.savefig(out_svg, format="svg"); plt.close()

def plot_and_save_pdp_grid_h2o(
    pipe, X_raw_df, top_features_transformed, out_svg,
    nbins=30, use_logodds=False, finite_only=True, top_k=5,
    n_ice=0,                    # >0 to overlay ICE
    x_quantile_span=(0.05,0.95),# crop x to central quantiles (None = no crop)
    ice_center=True,            # center ICE at reference x (median)
    y_from='pdp',               # 'pdp' -> y-lims from PDP mean (ignore ICE extremes)
    y_pad=0.05,                 # padding around PDP line for y-lims
    rug_max=800,                # max rug samples
    rug_color='k',              # black rug
    tick_color='k'              # black quantile ticks
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def _logit(p):
        p = np.clip(p, 1e-12, 1-1e-12)
        return np.log(p/(1-p))

    # choose up to top_k raw numeric features that exist in X
    feats = []
    for f in top_features_transformed:
        if f in X_raw_df.columns and pd.api.types.is_numeric_dtype(X_raw_df[f]):
            feats.append(f)
            if len(feats) == top_k:
                break
    if not feats:
        print("[PDP] No raw numeric features among top features; skipping.")
        return

    fig, axes = plt.subplots(1, len(feats), figsize=(5*len(feats), 4), constrained_layout=True)
    if len(feats) == 1:
        axes = [axes]

    rng = np.random.RandomState(0)

    for ax, feat in zip(axes, feats):
        # drop rows where this feature is not finite
        mask = np.isfinite(pd.to_numeric(X_raw_df[feat], errors="coerce")) if finite_only else np.ones(len(X_raw_df), bool)
        X_use = X_raw_df.loc[mask].copy()

        vals = pd.to_numeric(X_use[feat], errors="coerce").dropna().values
        if vals.size == 0:
            ax.text(0.5, 0.5, f"{feat}\n(no finite values)", ha="center", va="center")
            ax.axis("off"); continue

        # grid on central quantile span
        if x_quantile_span is None:
            qs = np.quantile(vals, np.linspace(0, 1, nbins))
        else:
            lo, hi = x_quantile_span
            qs = np.quantile(vals, np.linspace(lo, hi, nbins))
        grid = np.unique(qs)

        # quick Δ check using same quantile span
        loq, hiq = (x_quantile_span if x_quantile_span is not None else (0.05, 0.95))
        qlo, qhi = np.quantile(vals, [loq, hiq])
        X_lo = X_use.copy(); X_lo[feat] = qlo
        X_hi = X_use.copy(); X_hi[feat] = qhi
        p_lo = pipe.predict_proba(X_lo)[:, 1]
        p_hi = pipe.predict_proba(X_hi)[:, 1]
        y_lo = _logit(p_lo) if use_logodds else p_lo
        y_hi = _logit(p_hi) if use_logodds else p_hi
        print(f"[PDP-Δ] {feat}: mean@{int(100*loq)}%={np.mean(y_lo):.6f}, mean@{int(100*hiq)}%={np.mean(y_hi):.6f}, Δ={np.mean(y_hi)-np.mean(y_lo):.6f}")

        # PDP means
        means = []
        for v in grid:
            X_tmp = X_use.copy()
            X_tmp[feat] = v
            prob = pipe.predict_proba(X_tmp)[:, 1]
            yv = _logit(prob) if use_logodds else prob
            means.append(float(np.mean(yv)))
        means = np.array(means, dtype=float)
        line = ax.plot(grid, means, lw=2)[0]

        # Optional ICE
        if n_ice and len(X_use) > 0:
            rows = X_use.sample(n=min(n_ice, len(X_use)), random_state=0)
            # reference x for centering
            ref_x = np.median(vals)
            if ice_center:
                X_ref = rows.copy(); X_ref[feat] = ref_x
                y_ref = pipe.predict_proba(X_ref)[:, 1]
                if use_logodds: y_ref = _logit(y_ref)
            for i, (_, row) in enumerate(rows.iterrows()):
                X_row = pd.DataFrame([row.values]*len(grid), columns=rows.columns)
                X_row[feat] = grid
                y = pipe.predict_proba(X_row)[:, 1]
                if use_logodds: y = _logit(y)
                if ice_center:  y = y - y_ref[i]
                ax.plot(grid, y, lw=0.7, alpha=0.25)

        # x-limits to grid span (hides extreme outliers)
        ax.set_xlim(grid[0], grid[-1])

        # y-lims from PDP line so ICE doesn't flatten it
        y_min, y_max = float(np.nanmin(means)), float(np.nanmax(means))
        y_rng = max(1e-9, y_max - y_min)
        if y_from == 'pdp':
            ax.set_ylim(y_min - y_pad*y_rng, y_max + y_pad*y_rng)

        ymin, ymax = ax.get_ylim()

        # Rug (black)
        rug_y = ymin + 0.02*(ymax - ymin)
        if x_quantile_span is None:
            vals_for_rug = vals
        else:
            vals_for_rug = vals[(vals >= grid[0]) & (vals <= grid[-1])]
        if len(vals_for_rug):
            sample = pd.Series(vals_for_rug).sample(n=min(rug_max, len(vals_for_rug)), random_state=0).values
            ax.scatter(sample, np.full_like(sample, rug_y, dtype=float),
                       marker='|', alpha=0.4, s=80, linewidths=0, color=rug_color)

        # Quantile tick marks (black)
        tick_h = 0.02*(ymax - ymin)
        for q in np.unique(qs):
            ax.plot([q, q], [ymin, ymin + tick_h], lw=1, alpha=0.7, color=tick_color)

        ax.set_title(feat, fontsize=12)
        ax.set_xlabel("Feature Value", fontsize=10)
        ax.set_ylabel("Mean " + ("log-odds" if use_logodds else "probability"), fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Top 5 Partial Dependence Plots", fontsize=14)
    plt.savefig(out_svg, format="svg"); plt.close()


def shap_dependence_panels(X_raw, shap_values, feat_names, features,
                           nbins=30, qspan=(0.05,0.95), alpha=0.05, out_svg="SHAP_dep.svg",
                           dot_color=None):
    idx = {f:i for i,f in enumerate(feat_names)}
    n = len(features)
    ncols = (n + 1)//2 if n > 3 else n
    nrows = 2 if n > 3 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.4*nrows), constrained_layout=True)
    axes = np.array(axes).ravel()

    for ax, f in zip(axes, features):
        j = idx[f]
        x = pd.to_numeric(X_raw[f], errors="coerce")
        y = np.asarray(shap_values)[:, j]          # <-- ensure ndarray

        # make a boolean mask using numpy arrays on both sides
        m = x.notna().to_numpy() & np.isfinite(y)  # <-- Series -> ndarray for the mask

        # boolean-index both as numpy arrays (no .values on y)
        x = x.to_numpy()[m]                        # <-- was x[m].values
        y = y[m]                                   # <-- was y[m].values

        lo, hi = np.nanpercentile(x, [100*qspan[0], 100*qspan[1]])
        inr = (x >= lo) & (x <= hi)
        x_c, y_c = x[inr], y[inr]

        ax.scatter(x_c, y_c, s=4, alpha=0.12, rasterized=True, color=dot_color)

        qs = np.quantile(x_c, np.linspace(qspan[0], qspan[1], nbins))
        qs = np.unique(qs)
        mids, meds, p_lo, p_hi = [], [], [], []
        for a, b in zip(qs[:-1], qs[1:]):
            sel = (x_c >= a) & (x_c < b)
            if sel.sum() < 20:
                continue
            mids.append(0.5*(a+b))
            yy = np.sort(y_c[sel])
            meds.append(np.median(yy))
            p_lo.append(np.percentile(yy, 100*alpha))
            p_hi.append(np.percentile(yy, 100*(1-alpha)))

        if mids:
            ax.plot(mids, meds, lw=2)
            ax.fill_between(mids, p_lo, p_hi, alpha=0.2, linewidth=0)

        ax.set_xlim(lo, hi)
        ax.set_title(f)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP (log-odds)")
        ax.grid(True, ls="--", alpha=0.4)

    plt.savefig(out_svg, dpi=300)
    plt.close()

def hex_to_rgba(s):
    """Convert 8-digit hex (RRGGBBAA) or 6-digit hex (RRGGBB) to an RGBA tuple in [0,1].
    Returns None if s is falsy (e.g., None) so callers can skip coloring."""
    if not s:
        return None
    s = str(s).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = 1.0
        return (r, g, b, a)
    if len(s) == 8:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = int(s[6:8], 16) / 255.0
        return (r, g, b, a)
    return None

def main():

    # ------------------------------ #
    # Debug / speed toggles
    # ------------------------------ #
    SPEED_MODE       = False    # Balanced sample + stratified CV (fast iterations)
    RUN_BIOME_SHAP   = True   # Turn on when you want biome SHAP again
    RUN_ARIDITY_SHAP = True 
    RUN_ABLATIONS    = True   # Turn on when you want ablations again
    MAX_SHAP_ROWS    = 5000    # Global SHAP rows (2000 default)
    BIOME_SHAP_MAX   = 500     # Max sampled points per biome (200 default)
    BIOME_TOP_FEATS  = 15      # <-- plot top 15 features in SHAP-by-biome

    # ------------------------------ #
    # CLI args (unchanged)
    # ------------------------------ #
    parser = argparse.ArgumentParser(description="EWS + Env ML classifier")
    parser.add_argument("--tau_full", type=str, help="Zarr path to full tau dataset (KT).")
    parser.add_argument("--tau_pre", type=str, help="Zarr path to pre-break tau dataset for POSITIVES (KT).")
    parser.add_argument("--tau_pre_neg", type=str, help="Zarr path to pre-break tau dataset for NEGATIVES (KT).")
    parser.add_argument("--breaks", type=str, help="Zarr path to changepoint dataset (choose fields via --cp-test).")
    parser.add_argument("--var", type=str, choices=["sm","Et","precip"], help="Variable prefix for tau columns and preset predictors.")
    parser.add_argument("--cp_test", type=str, choices=["pettitt","stc","var"], help="Which breakpoint fields to use: pettitt, stc, var (should match the path to the breaks ds).")
    parser.add_argument("--outdir", type=str, help="Base output directory. A subfolder per variable/test will be created.")
    parser.add_argument("--pos-n", type=int, default=2000, help="Positive class sample size.")
    parser.add_argument("--neg-n", type=int, default=2000, help="Negative class sample size.")
    parser.add_argument("--alpha", type=float, default=0.05, help="P-value threshold for tau significance.")
    args = parser.parse_args()

    cp_key = args.cp_test.lower()
    outdir = Path(args.outdir) / f"{args.var}_{cp_key}_SCIPY"
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------ #
    # Predictors (unchanged lists)
    # ------------------------------ #
    predictors_map = {
        "sm": [
            "/home/romi/ews/data/driver_analysis_final/driver_temperature.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_precipitation.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_soil_moisture.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_transpiration.zarr",
            "/home/romi/ews/data/driver_analysis/driver_pet.zarr",
            "/home/romi/ews/data/driver_analysis/driver_aridity.zarr",
            "/home/romi/ews/data/driver_analysis/driver_groundwater_table.zarr",
            "/home/romi/ews/data/driver_analysis/driver_GPP.zarr",
            "/home/romi/ews/data/driver_analysis/driver_tree_cover.zarr",
            "/home/romi/ews/data/driver_analysis/driver_non_tree_cover.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_enso.zarr",
            # "/home/romi/ews/data/driver_analysis/global_irrigated_areas.zarr",
            "/home/romi/ews/data/driver_analysis/driver_crop_cover.zarr",
        ],
        "Et": [
            "/home/romi/ews/data/driver_analysis_final/driver_temperature.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_precipitation.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_soil_moisture.zarr",
            "/home/romi/ews/data//driver_analysis_final/driver_transpiration.zarr",
            "/home/romi/ews/data/driver_analysis/driver_pet.zarr",
            "/home/romi/ews/data/driver_analysis/driver_aridity.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_albedo.zarr",
            "/mnt/data/romi/data/driver_analysis_final/driver_boundary_layer_height.zarr",
            "/home/romi/ews/data/driver_analysis/driver_GPP.zarr",
            "/home/romi/ews/data/driver_analysis/driver_tree_cover.zarr",
            "/home/romi/ews/data/driver_analysis/driver_non_tree_cover.zarr",
            # "/home/romi/ews/data/driver_analysis/global_irrigated_areas.zarr",
            "/home/romi/ews/data/driver_analysis/driver_crop_cover.zarr",
            "/home/romi/ews/data/driver_analysis/driver_vpd.zarr",
            "/home/romi/ews/data/driver_analysis/driver_groundwater_table.zarr",
        ],
        "precip": [
            "/home/romi/ews/data/driver_analysis_final/driver_temperature.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_precipitation.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_soil_moisture.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_transpiration.zarr",
            "/home/romi/ews/data/driver_analysis/driver_pet.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_transpiration.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_cape.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_boundary_layer_height.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_albedo.zarr",
            "/home/romi/ews/data/driver_analysis/driver_GPP.zarr",
            "/home/romi/ews/data/driver_analysis/driver_tree_cover.zarr",
            "/home/romi/ews/data/driver_analysis/driver_non_tree_cover.zarr",
            # "/home/romi/ews/data/driver_analysis/global_irrigated_areas.zarr",
            "/home/romi/ews/data/driver_analysis/driver_crop_cover.zarr",
            "/home/romi/ews/data/driver_analysis_final/driver_enso.zarr",
        ],
    }

    vars_to_drop = [
        "step", "degree", "spatial_ref", "number", "surface",
        "depthBelowLandLayer", "PDSI_spatial_ref", "GPP_spatial_ref",
        "tree_cover_spatial_ref", "non_tree_cover_spatial_ref", "elevation_spatial_ref",
        "soil_texture_processed_spatial_ref", "enso_spatial_ref", "lulcc_spatial_ref",
        "crop_cover_band_data", "crop_cover_spatial_ref", "band", "lulcc_change",
        "number_x", "time_x", "step_x", "surface_x", "valid_time_x", "valid_time_x",
        "spatial_ref_x", "number_y", "time_y", "step_y", "surface_y", "valid_time_y",
        "spatial_ref_y", "valid_time", "time", "latitude_x", "longitude_x", "latitude_y",
        "longitude_y"
    ]

    # ------------------------------ #
    # Build target + predictors
    # ------------------------------ #
    df_all, ds_tau = build_target_dataframe(
        args.tau_full, args.tau_pre, args.breaks, args.var, cp_key,
        alpha=args.alpha, tau_pre_neg_path=args.tau_pre_neg
    )

    predictors = []
    tgt_lat = ds_tau["lat"]; tgt_lon = ds_tau["lon"]
    for path in predictors_map[args.var]:
        if not os.path.exists(path):
            print(f"Predictor missing, skipping: {path}")
            continue
        ds_pred = load_and_harmonize_predictor(path, tgt_lat, tgt_lon, vars_to_drop)
        predictors.append(ds_pred)
    if len(predictors) == 0:
        print("No predictors loaded. Exiting."); sys.exit(1)

    ds_predictors = xr.merge(predictors)
    df_predictors = ds_predictors.to_dataframe().reset_index()

    base_bad = {"valid_time","time","step","number","surface","spatial_ref","degree","band"} | set(vars_to_drop)
    def _base(name): return name[:-2] if name.endswith(("_x","_y")) else name
    cols_to_drop = [c for c in df_predictors.columns if c not in ("lat","lon") and (_base(c) in base_bad or c in base_bad)]
    if cols_to_drop:
        df_predictors.drop(columns=[c for c in cols_to_drop if c in df_predictors.columns], inplace=True, errors="ignore")
    const_cols = [c for c in df_predictors.columns if c not in ("lat","lon") and df_predictors[c].nunique(dropna=False) <= 1]
    if const_cols:
        df_predictors.drop(columns=const_cols, inplace=True)

    # Categorical handling flags
    st_cols = [c for c in df_predictors.columns if c.endswith("soil_texture_processed_mean")]
    for col in st_cols:
        df_predictors[col] = df_predictors[col].astype("category")
    ira_cols = [c for c in df_predictors.columns if c.endswith("global_irrigated_areas_irrigated_area")]
    for col in ira_cols:
        df_predictors[col] = (df_predictors[col] > 0).astype(int)

    # Merge target + predictors
    df_all = df_all.merge(df_predictors, on=["lat", "lon"], how="left")

    df_all  = df_all [df_all ["lat"] > -60.0].copy()

    # KT columns sanity
    ews_cols_all = [c for c in df_all.columns if c.endswith("_kt")]
    print("Example KT stats:", df_all[ews_cols_all].describe(percentiles=[.01,.5,.99]).T.head(10))

    # ------------------------------ #
    # Final numeric coercion
    # ------------------------------ #
    df_model = df_all.copy()
    ews_cols = [c for c in df_model.columns if c.endswith("_kt")]
    numeric_candidates = [c for c in df_model.columns if c not in (["lat","lon","target","fold_id"] + st_cols)]
    df_model[ews_cols] = df_model[ews_cols].apply(pd.to_numeric, errors="coerce")
    df_model[numeric_candidates] = df_model[numeric_candidates].apply(pd.to_numeric, errors="coerce")
    df_model = df_model[df_model["lat"] > -60.0].copy()

    # Features + correlation
    non_feature = {"lat","lon","target","fold_id"} | set(vars_to_drop)
    feature_cols = [c for c in df_model.columns if c not in non_feature and pd.api.types.is_numeric_dtype(df_model[c])]
    plot_and_save_corr_heatmap(df_model, feature_cols, str(outdir / f"{args.var}_corr_heatmap.svg"))


    # ------------------------------ #
    # Speed mode vs spatial folds
    # ------------------------------ #
    if SPEED_MODE:
        df_model = make_balanced_sample(df_model, pos_n=args.pos_n, neg_n=args.neg_n, seed=RANDOM_STATE)
        USE_FOLD_COLUMN = False
    else:
        USE_FOLD_COLUMN = True
        df_model["lat_bin"] = (df_model["lat"] // 10).astype(int)
        df_model["lon_bin"] = (df_model["lon"] // 10).astype(int)
        df_model["block"]   = df_model["lat_bin"].astype(str) + "_" + df_model["lon_bin"].astype(str)
        K = 5
        rng = np.random.RandomState(RANDOM_STATE)
        blocks = df_model["block"].unique()
        rng.shuffle(blocks)
        block_to_fold = {b: i % K for i, b in enumerate(blocks)}
        df_model["fold_id"] = df_model["block"].map(block_to_fold).astype(int)
        

    # *_kt* coercion range check
    def _coerce_float01(df, cols):
        bad = []
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.astype("float32")
            if s.abs().max(skipna=True) > 1.0001:
                bad.append((c, float(s.min(skipna=True)), float(s.max(skipna=True))))
        return bad
    kt_out_of_range = _coerce_float01(df_model, ews_cols)
    print("KT columns out of expected range:", kt_out_of_range[:10], "… total:", len(kt_out_of_range))

    # ------------------------------ #
    # Build X/y and preprocessing
    # ------------------------------ #
    y = df_model["target"].astype(int).values
    X = df_model[feature_cols].copy()

    # Identify categoricals to one-hot (soil texture)
    cat_features = [c for c in st_cols if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat_features),
            ("num", "passthrough", num_features),
        ],
        remainder="drop"
    )

    # Base XGB params (roughly matched to your H2O config)
    base_xgb_params = dict(
        n_estimators=300,
        max_depth=8,
        min_child_weight=10,
        gamma=1.0,
        reg_lambda=5.0,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=4
    )

    # Helper to fit across folds, gather pooled predictions and per-fold metrics
    def cv_fit_predict(X_df, y_vec, fold_ids=None, label="Full"):
        if fold_ids is None:
            # StratifiedKFold for SPEED_MODE path
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            splits = list(skf.split(X_df, y_vec))
        else:
            # PredefinedSplit from fold_id
            test_fold = np.array(fold_ids, dtype=int)
            ps = PredefinedSplit(test_fold=test_fold)
            # Build splits explicitly
            splits = []
            for f in np.unique(test_fold):
                test_idx = np.where(test_fold == f)[0]
                train_idx = np.where(test_fold != f)[0]
                splits.append((train_idx, test_idx))

        pooled_prob = np.full(y_vec.shape[0], np.nan, dtype=float)
        rows = []

        # We create pipeline fresh each fold (so OneHot categories fit on train only)
        for fnum, (tr, te) in enumerate(splits):
            # scale_pos_weight per-fold (neg/pos)
            pos_tr = y_vec[tr].sum()
            neg_tr = len(tr) - pos_tr
            spw = float(neg_tr / max(1, pos_tr)) if pos_tr > 0 else 1.0

            xgb = XGBClassifier(**base_xgb_params, scale_pos_weight=spw)
            pipe = Pipeline([("pre", preproc), ("clf", xgb)])

            pipe.fit(X_df.iloc[tr], y_vec[tr])
            prob = pipe.predict_proba(X_df.iloc[te])[:, 1]
            pooled_prob[te] = prob

            # Per-fold metrics
            yf, pf = y_vec[te], prob
            if np.unique(yf).size >= 2:
                prf = average_precision_score(yf, pf)
                rocf = roc_auc_score(yf, pf)
            else:
                prf = np.nan; rocf = np.nan
            rows.append({
                "model": label,
                "fold": int(fnum if fold_ids is None else np.unique(fold_ids)[fnum]),
                "n": int(len(te)),
                "positives": int(yf.sum()),
                "pr_auc": float(prf),
                "roc_auc": float(rocf),
            })

        # Micro (pooled) metrics
        ap_micro = float(average_precision_score(y_vec, pooled_prob))
        roc_micro = float(roc_auc_score(y_vec, pooled_prob))

        # Macro averages
        fold_df = pd.DataFrame(rows)
        ap_macro_unw = float(np.nanmean(fold_df["pr_auc"])) if len(fold_df) else np.nan
        roc_macro_unw = float(np.nanmean(fold_df["roc_auc"])) if len(fold_df) else np.nan

        # Weighted by fold size
        w_size = fold_df["n"].to_numpy(dtype=float)
        if w_size.sum() > 0:
            ap_macro_w_size = float(np.nansum(w_size * fold_df["pr_auc"]) / w_size.sum())
            roc_macro_w_size = float(np.nansum(w_size * fold_df["roc_auc"]) / w_size.sum())
        else:
            ap_macro_w_size = np.nan; roc_macro_w_size = np.nan

        # Weighted by #positives (for PR AUC)
        w_pos = fold_df["positives"].to_numpy(dtype=float)
        if w_pos.sum() > 0:
            ap_macro_w_pos = float(np.nansum(w_pos * fold_df["pr_auc"]) / w_pos.sum())
        else:
            ap_macro_w_pos = np.nan

        base_ap = float(np.mean(y_vec))
        ap_micro = float(average_precision_score(y_vec, pooled_prob))
        roc_micro = float(roc_auc_score(y_vec, pooled_prob))

        ap_lift = float(ap_micro / base_ap) if base_ap > 0 else np.nan
        ap_gain = float((ap_micro - base_ap) / (1 - base_ap)) if base_ap < 1 else np.nan

        summ = pd.DataFrame([{
            "model": label,
            "PR_AUC_micro": ap_micro,
            "ROC_AUC_micro": roc_micro,
            "PR_AUC_macro_unweighted": ap_macro_unw,
            "PR_AUC_macro_w_size": ap_macro_w_size,
            "PR_AUC_macro_w_pos": ap_macro_w_pos,
            "ROC_AUC_macro_unweighted": roc_macro_unw,
            "ROC_AUC_macro_w_size": roc_macro_w_size,
            "baseline_ap": base_ap,
            "ap_lift_x": ap_lift,
            "ap_gain_normalized": ap_gain,
            "total_samples": int(len(y_vec)),
            "total_positives": int(y_vec.sum())
        }])

        return pooled_prob, fold_df, summ

    # Fit CV (main model)
    fold_ids = df_model["fold_id"].values if USE_FOLD_COLUMN else None
    y_prob, fold_df, summ_df = cv_fit_predict(X, y, fold_ids=fold_ids, label="Full")
    # Save fold metrics
    if len(fold_df):
        fold_df.to_csv(outdir / f"{args.var}_fold_metrics.csv", index=False)
    # Save summary
    summ_df.to_csv(outdir / f"{args.var}_metrics_summary.csv", index=False)

    # PR / ROC plots from pooled predictions
    prec, rec, _ = precision_recall_curve(y, y_prob)
    pr_auc_overall = float(average_precision_score(y, y_prob))
    roc_auc_overall = float(roc_auc_score(y, y_prob))
    plt.figure(figsize=(6, 5)); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={pr_auc_overall:.3f})")
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(outdir / f"{args.var}_precision_recall_curve.svg", format="svg"); plt.close()

    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_overall:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC curve"); plt.legend(); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(outdir / f"{args.var}_roc_curve.svg", format="svg"); plt.close()

    # Calibration
    plot_and_save_calibration(y, y_prob, outdir / f"{args.var}_calibration.svg")

    # --------------------------------------------------- #
    # Train final model on all data (for SHAP/PDP/varimp)
    # --------------------------------------------------- #
    # Compute spw on full data (for comparability )
    pos_all = int(y.sum()); neg_all = int(len(y) - y.sum())
    spw_all = float(neg_all / max(1, pos_all)) if pos_all > 0 else 1.0

    final_xgb = XGBClassifier(**base_xgb_params, scale_pos_weight=spw_all)
    final_pipe = Pipeline([("pre", preproc), ("clf", final_xgb)])
    final_pipe.fit(X, y)

    # Feature names after preprocessing
    ohe = final_pipe.named_steps["pre"].named_transformers_["cat"]
    cat_out = []
    if cat_features:
        cat_out = list(ohe.get_feature_names_out(cat_features))
    feat_names = cat_out + num_features

    # Var importance
    importances = final_pipe.named_steps["clf"].feature_importances_
    varimp_df = pd.DataFrame({"variable": feat_names, "importance": importances})
    varimp_df.to_csv(outdir / f"{args.var}_varimp_full.csv", index=False)
    plot_and_save_varimp(varimp_df, str(outdir / "varimp_top10.svg"), top_n=10)

    # PDP (top 5)
    # After you compute varimp_df, num_features, and X:
    print("[DEBUG] First 15 varimp features:", varimp_df.sort_values("importance", ascending=False)["variable"].head(15).tolist())
    print("[DEBUG] # of *_kt columns in X:", sum(c.endswith("_kt") for c in X.columns))
    print("[DEBUG] Top-10 *_kt by varimp:", (varimp_df[varimp_df["variable"].str.endswith("_kt")]
                                            .sort_values("importance", ascending=False)["variable"].head(10).tolist()))
    for c in [f for f in X.columns if f.endswith("_kt")][:5]:
        s = pd.to_numeric(X[c], errors="coerce")
        print(f"[DEBUG] {c}: n={s.size}, NaN%={s.isna().mean():.3f}, nunique={s.nunique(dropna=True)}")
        
        
    # Overall ranking
    top_all = varimp_df.sort_values("importance", ascending=False)["variable"].tolist()

    # Top-5 _kt
    top_kt = (varimp_df[varimp_df["variable"].str.endswith("_kt")]
            .sort_values("importance", ascending=False)["variable"]
            .tolist()[:5])

    # Overall PDPs
    plot_and_save_pdp_grid_h2o(
        final_pipe, X, top_all,
        out_svg=str(outdir / f"{args.var}_pdp_top5_overall.svg"),
        nbins=30, use_logodds=False, n_ice=0, x_quantile_span=(0.05,0.95)
    )

    plot_and_save_pdp_grid_h2o(
        final_pipe, X, top_kt,
        out_svg=str(outdir / f"{args.var}_pdp_top5_kt.svg"),
        nbins=30, use_logodds=False, n_ice=0,
        y_from='pdp', x_quantile_span=(0.05,0.95)
    )

    # ------------------------------ #
    # SHAP 
    # ------------------------------ #
    # Limit rows for SHAP to control memory
    n_shap = min(MAX_SHAP_ROWS, len(X))
    rng = np.random.RandomState(0)
    shap_idx = np.sort(rng.choice(len(X), size=n_shap, replace=False))
    X_trans = final_pipe.named_steps["pre"].transform(X.iloc[shap_idx])
    explainer = shap.TreeExplainer(final_pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_trans)
    base_raw    = float(explainer.expected_value) 
    base_prob   = 1 / (1 + np.exp(-base_raw))

    with open(outdir / f"{args.var}_shap_baseline.txt", "w") as f:
        f.write(f"base_log_odds={base_raw:.6f}\nbase_probability={base_prob:.6f}\n")

    # SHAP summary (mean |shap|)
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
    shap_signed_mean = np.mean(shap_values, axis=0)
    shap_summary = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": shap_abs_mean,
        "mean_signed_shap": shap_signed_mean
    }).sort_values("mean_abs_shap", ascending=False)
    shap_summary.to_csv(outdir / f"{args.var}_shap_summary_signed.csv", index=False)

    # SHAP bar
    topN = 20
    top_imp = shap_summary.head(topN)
    plt.figure(figsize=(8, 5.5))
    plt.barh(top_imp["feature"][::-1], top_imp["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP| (global importance)"); plt.ylabel("Feature")
    plt.title(f"SHAP global importance — Top {topN}")
    plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_importance_top{topN}.svg", format="svg"); plt.close()

    # SHAP beeswarm (requires raw matrix)
    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X_trans, feature_names=feat_names, show=False, plot_size=None)
    plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_beeswarm_top{topN}.svg", format="svg"); plt.close()

    var_dot_hex_map = {
        "Et":     "#34692933",  # dark green
        "sm":     "#635b2733",  # dark brown
        "precip": "#003849ff"   # dark blue
    }
    dot_color = hex_to_rgba(var_dot_hex_map.get(args.var, None))

    X_raw_for_dep = X.iloc[shap_idx].copy()  # align rows with shap_values
    top_numeric = [f for f in shap_summary["feature"].tolist() if f in num_features][:6]
    if top_numeric:
        shap_dependence_panels(
            X_raw_for_dep, shap_values, feat_names, top_numeric,
            nbins=30, qspan=(0.05, 0.95), alpha=0.05,
            out_svg=outdir / f"{args.var}_shap_dependence_top{len(top_numeric)}.svg",
            dot_color=dot_color
        )

    # ------------------------------ #
    # SHAP by biome 
    # ------------------------------ #
    if RUN_BIOME_SHAP:
        biomes_fp = "/home/romi/ews/data/terr-ecoregions-TNC/tnc_terr_ecoregions.shp"

        biomes_name_map = {
            'Boreal Forests/Taiga': 'Boreal Forests/Taiga',
            'Deserts and Xeric Shrublands': 'Deserts and Xeric Shrublands',
            'Mediterranean Forests, Woodlands and Scrub': 'Mediterranean Forests',
            'Montane Grasslands and Shrublands': 'Montane Grasslands',
            'Temperate Broadleaf and Mixed Forests': 'Temperate Broadleaf Forests',
            'Temperate Conifer Forests': 'Temperate Conifer Forests',
            'Temperate Grasslands, Savannas and Shrublands': 'Temperate Grasslands',
            'Tropical and Subtropical Coniferous Forests': 'Tropical Coniferous Forests',
            'Tropical and Subtropical Dry Broadleaf Forests': 'Tropical Dry Forests',
            'Tropical and Subtropical Grasslands, Savannas and Shrublands': 'Tropical Grasslands',
            'Tropical and Subtropical Moist Broadleaf Forests': 'Tropical Moist Forests',
            'Tundra': 'Tundra'
        }

        ordered_biomes = [
            'Tropical Moist Forests','Tropical Grasslands','Tropical Dry Forests','Tropical Coniferous Forests',
            'Temperate Broadleaf Forests','Temperate Conifer Forests','Temperate Grasslands','Montane Grasslands',
            'Mediterranean Forests','Deserts and Xeric Shrublands','Boreal Forests/Taiga','Tundra'
        ]

        df_for_shap = df_model[["lat", "lon", "target"] + [c for c in feature_cols if c in df_model.columns]].copy()
        df_for_shap["geometry"] = [Point(xy) for xy in zip(df_for_shap["lon"], df_for_shap["lat"])]
        gdf = gpd.GeoDataFrame(df_for_shap, geometry="geometry", crs="EPSG:4326")

        biomes = gpd.read_file(biomes_fp)
        if biomes.crs is None or biomes.crs.to_string().upper() != "EPSG:4326":
            biomes = biomes.to_crs("EPSG:4326")
        try:
            biomes = biomes.set_geometry(biomes.geometry.buffer(0))
        except Exception:
            pass

        g_join = gpd.sjoin(gdf, biomes[["WWF_MHTNAM","geometry"]], how="left", predicate="intersects")
        g_join["WWF_MHTNAM"] = g_join["WWF_MHTNAM"].fillna("Unknown").replace(biomes_name_map)
        drop_biomes = {"Mangroves","Inland Water","Rock and Ice","Flooded Grasslands and Savannas","Unknown"}
        g_join = g_join[~g_join["WWF_MHTNAM"].isin(drop_biomes)].copy()

        biome_shap_max = int(max(1, BIOME_SHAP_MAX))
        sampled = (g_join.groupby("WWF_MHTNAM", group_keys=False)
                         .apply(lambda d: d.sample(min(len(d), biome_shap_max),
                                                random_state=RANDOM_STATE))
                         .reset_index(drop=True))
        sampled["biome"] = sampled["WWF_MHTNAM"].astype(str)
        
        # Transform sampled X to the final feature space
        X_sample       = sampled[feature_cols]
        X_sample_trans = final_pipe.named_steps["pre"].transform(X_sample)
        sv             = explainer.shap_values(X_sample_trans)

        # Aggregate per biome
        mean_signed = {}
        mean_abs = {}
        for b, d in sampled.groupby("biome"):
            pos = d.index.to_numpy() 
            sv_b = sv[pos, :]        
            mean_signed[b] = sv_b.mean(axis=0)
            mean_abs[b]    = np.abs(sv_b).mean(axis=0)

        mean_signed_df = pd.DataFrame(mean_signed, index=feat_names).T
        mean_abs_df    = pd.DataFrame(mean_abs, index=feat_names).T

        # Order features by global |SHAP|
        
        order_feats = shap_summary["feature"].tolist()[:BIOME_TOP_FEATS]
        ms = mean_signed_df[order_feats]
        ma = mean_abs_df[order_feats]

        # Rename biomes
        ms.index = [biomes_name_map.get(b, b) for b in ms.index]
        ma.index = [biomes_name_map.get(b, b) for b in ma.index]

        ordered_biomes = [
            'Tropical Moist Forests','Tropical Grasslands','Tropical Dry Forests','Tropical Coniferous Forests',
            'Temperate Broadleaf Forests','Temperate Conifer Forests','Temperate Grasslands','Montane Grasslands',
            'Mediterranean Forests','Deserts and Xeric Shrublands','Boreal Forests/Taiga','Tundra'
        ]

        present = [b for b in ordered_biomes if b in ms.index]
        extras  = [b for b in ms.index if b not in ordered_biomes]

        ms = ms.loc[present + extras]
        ma = ma.loc[present + extras]

        # Save + heatmaps (features as rows)
        n_avail = g_join["WWF_MHTNAM"].value_counts()
        n_used  = sampled["WWF_MHTNAM"].value_counts()

        n_avail.index = [biomes_name_map.get(b, b) for b in n_avail.index]
        n_used.index  = [biomes_name_map.get(b, b) for b in n_used.index]

        cols = list(ms.index)
        counts_block = pd.DataFrame(
            [n_avail.reindex(cols), n_used.reindex(cols)],
            index=["N_available", "N_used"]
            )
        
        ms_out = pd.concat([ms.T, counts_block], axis=0)
        ma_out = pd.concat([ma.T, counts_block], axis=0)

        ms_out.to_csv(outdir / f"{args.var}_shap_by_biome_mean_signed_top{BIOME_TOP_FEATS}.csv")
        ma_out.to_csv(outdir / f"{args.var}_shap_by_biome_mean_abs_top{BIOME_TOP_FEATS}.csv")

        vmax = np.nanmax(np.abs(ms.T.values))
        plt.figure(figsize=(0.6*len(ms.index)+4, 0.35*len(ms.columns)+2))
        sn.heatmap(ms.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, center=0, annot=False)
        plt.title(f"{args.var} — SHAP (signed) by biome (≤{biome_shap_max}/biome) — Top {BIOME_TOP_FEATS}")
        plt.ylabel("Feature"); plt.xlabel("Biome")
        plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_by_biome_heatmap_signed_top{BIOME_TOP_FEATS}.svg", dpi=300)
        plt.close()

        var_dot_hex_map = {
            "Et":     "#2c5e45ff",  # dark green
            "sm":     "#5e5a2cff",  # dark brown
            "precip": "#003849ff"   # dark blue
        }
        hexcode = var_dot_hex_map.get(args.var, "#003849ff")
        target_rgb = mcolors.to_rgb(hexcode)
        cmap_abs = LinearSegmentedColormap.from_list("white_to_var", [(1,1,1), target_rgb], N=256)

        vmax_abs = float(np.nanmax(ma.values)) if np.isfinite(ma.values).any() else 1.0

        plt.figure(figsize=(0.6*len(ma.index)+4, 0.35*len(ma.columns)+2))
        sn.heatmap(ma.T, cmap=cmap_abs, vmin=0,vmax = vmax_abs, annot=False)  
        plt.title(f"{args.var} — SHAP (abs) by biome (≤{biome_shap_max}/biome) — Top {BIOME_TOP_FEATS}")
        plt.ylabel("Feature"); plt.xlabel("Biome")
        plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_by_biome_heatmap_abs_top{BIOME_TOP_FEATS}.svg", dpi=300)
        plt.close()


    # ------------------------------ #
    # SHAP by aridity class 
    # ------------------------------ #
    
    if RUN_ARIDITY_SHAP:
        BIOME_TOP_FEATS = 10 # just 10 for aridity so it can be a square
        # 1) Compute aridity class on the same grid as ds_tau, then join to samples
        arid_da = compute_aridity_classes_like(ds_tau)  # returns DataArray named 'aridity_class'
        arid_df = arid_da.to_dataframe(name="aridity_class").reset_index()

        df_ar = df_model.merge(arid_df, on=["lat", "lon"], how="left")
        df_ar = df_ar[df_ar["aridity_class"].notna()].copy()
        df_ar["aridity_class"] = df_ar["aridity_class"].astype(str)

        # 2) Prepare SHAP frame (same features as elsewhere)
        df_for_shap = df_ar[["lat", "lon", "target", "aridity_class"] + [c for c in feature_cols if c in df_ar.columns]].copy()

        # 3) Sample up to BIOME_SHAP_MAX per aridity class (reuse the same cap)
        class_shap_max = int(max(1, BIOME_SHAP_MAX))
        sampled = (df_for_shap.groupby("aridity_class", group_keys=False)
                           .apply(lambda d: d.sample(min(len(d), class_shap_max),
                                                     random_state=RANDOM_STATE))
                           .reset_index(drop=True))
        sampled["aridity"] = sampled["aridity_class"].astype(str)

        # 4) Transform to model feature space and compute SHAP on the sampled set
        X_sample       = sampled[feature_cols]
        X_sample_trans = final_pipe.named_steps["pre"].transform(X_sample)
        sv             = explainer.shap_values(X_sample_trans)

        # 5) Aggregate per aridity class
        mean_signed = {}
        mean_abs    = {}
        for a, d in sampled.groupby("aridity"):
            pos = d.index.to_numpy()          # indices are 0..N-1 after reset_index(drop=True)
            sv_a = sv[pos, :]
            mean_signed[a] = sv_a.mean(axis=0)
            mean_abs[a]    = np.abs(sv_a).mean(axis=0)

        mean_signed_df = pd.DataFrame(mean_signed, index=feat_names).T
        mean_abs_df    = pd.DataFrame(mean_abs,    index=feat_names).T

        # 6) Keep the same global top features ordering
        order_feats = shap_summary["feature"].tolist()[:BIOME_TOP_FEATS]
        ms = mean_signed_df[order_feats]
        ma = mean_abs_df[order_feats]

        # Optional: if ARIDITY_LABELS exists globally, use it to order classes
        if "ARIDITY_LABELS" in globals():
            ordered = [c for c in ARIDITY_LABELS if c in ms.index]
            extras  = [c for c in ms.index if c not in ordered]
            if ordered:    # only reindex when we actually matched labels
                ms = ms.loc[ordered + extras]
                ma = ma.loc[ordered + extras]

        # 7) Save CSVs with N_available and N_used appended (same pattern as biome)
        n_avail = df_ar["aridity_class"].value_counts()
        n_used  = sampled["aridity_class"].value_counts()
        cols = list(ms.index)
        counts_block = pd.DataFrame(
            [n_avail.reindex(cols), n_used.reindex(cols)],
            index=["N_available", "N_used"]
        )
        ms_out = pd.concat([ms.T, counts_block], axis=0)
        ma_out = pd.concat([ma.T, counts_block], axis=0)

        ms_out.to_csv(outdir / f"{args.var}_shap_by_aridity_mean_signed_top{BIOME_TOP_FEATS}.csv")
        ma_out.to_csv(outdir / f"{args.var}_shap_by_aridity_mean_abs_top{BIOME_TOP_FEATS}.csv")

        # 8) Heatmaps (same styling, filenames include 'aridity')
        vmax = np.nanmax(np.abs(ms.T.values))
        plt.figure(figsize=(0.6*len(ms.index)+4, 0.35*len(ms.columns)+2))
        sn.heatmap(ms.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, center=0, annot=False)
        plt.title(f"{args.var} — SHAP (signed) by aridity class (≤{class_shap_max}/class) — Top {BIOME_TOP_FEATS}")
        plt.ylabel("Feature"); plt.xlabel("Aridity class")
        plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_by_aridity_heatmap_signed_top{BIOME_TOP_FEATS}.svg", dpi=300)
        plt.close()

        var_dot_hex_map = {
            "Et":     "#2c5e45ff",  # dark green
            "sm":     "#5e5a2cff",  # dark brown
            "precip": "#003849ff"   # dark blue
        }
        hexcode = var_dot_hex_map.get(args.var, "#003849ff")
        target_rgb = mcolors.to_rgb(hexcode)
        cmap_abs = LinearSegmentedColormap.from_list("white_to_var", [(1,1,1), target_rgb], N=256)

        vmax_abs = float(np.nanmax(ma.values)) if np.isfinite(ma.values).any() else 1.0

        plt.figure(figsize=(0.6*len(ma.index)+4, 0.35*len(ma.columns)+2))
        sn.heatmap(ma.T, cmap=cmap_abs, vmin=0, vmax = vmax_abs, annot=False)
        plt.title(f"{args.var} — SHAP (abs) by aridity class (≤{class_shap_max}/class) — Top {BIOME_TOP_FEATS}")
        plt.ylabel("Feature"); plt.xlabel("Aridity class")
        plt.tight_layout(); plt.savefig(outdir / f"{args.var}_shap_by_aridity_heatmap_abs_top{BIOME_TOP_FEATS}.svg", dpi=300)
        plt.close()

    # ------------------------------ #
    # Ablations
    # ------------------------------ #
    if RUN_ABLATIONS:
        def run_abl(feats, label):
            X_sub = df_model[feats].copy()
            # rebuild cat/num per subset
            cat_sub = [c for c in cat_features if c in feats]
            num_sub = [c for c in X_sub.columns if c not in cat_sub]
            pre = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat_sub),
                 ("num", "passthrough", num_sub)]
            )
            # Replace preproc inside cv loop by closure
            nonlocal_preproc = pre

            def cv_local(Xd, yd, fold_ids_local):
                # as above but uses local preproc
                if fold_ids_local is None:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    splits = list(skf.split(Xd, yd))
                else:
                    test_fold = np.array(fold_ids_local, dtype=int)
                    splits = []
                    for f in np.unique(test_fold):
                        te = np.where(test_fold == f)[0]
                        tr = np.where(test_fold != f)[0]
                        splits.append((tr, te))

                pooled = np.full(yd.shape[0], np.nan)
                rows = []
                for fnum, (tr, te) in enumerate(splits):
                    pos_tr = yd[tr].sum(); neg_tr = len(tr) - pos_tr
                    spw = float(neg_tr / max(1, pos_tr)) if pos_tr > 0 else 1.0
                    xgb = XGBClassifier(**base_xgb_params, scale_pos_weight=spw)
                    pipe = Pipeline([("pre", nonlocal_preproc), ("clf", xgb)])
                    pipe.fit(Xd.iloc[tr], yd[tr])
                    pr = pipe.predict_proba(Xd.iloc[te])[:, 1]
                    pooled[te] = pr
                    yf, pf = yd[te], pr
                    if np.unique(yf).size >= 2:
                        prf = average_precision_score(yf, pf); rocf = roc_auc_score(yf, pf)
                    else:
                        prf = np.nan; rocf = np.nan
                    rows.append({"model": label, "fold": int(fnum), "n": int(len(te)),
                                 "positives": int(yf.sum()), "pr_auc": float(prf), "roc_auc": float(rocf)})
                fold = pd.DataFrame(rows)
                ap_micro = float(average_precision_score(yd, pooled))
                roc_micro = float(roc_auc_score(yd, pooled))
                summ = pd.DataFrame([{"model": label, "PR_AUC_micro": ap_micro, "ROC_AUC_micro": roc_micro}])
                return fold, summ

            return cv_local(X_sub, y, fold_ids)

        ews_feats    = [c for c in feature_cols if c.endswith("_kt")]
        driver_feats = [c for c in feature_cols if c not in ews_feats]

        folds_ews,  summ_ews  = run_abl(ews_feats,    "EWS-only")
        folds_drv,  summ_drv  = run_abl(driver_feats, "Drivers-only")
        folds_full, summ_full = run_abl(feature_cols, "Full")

        if not (folds_ews.empty and folds_drv.empty and folds_full.empty):
            pd.concat([folds_ews, folds_drv, folds_full], ignore_index=True)\
              .to_csv(outdir / f"{args.var}_ablations_fold_metrics.csv", index=False)
        pd.concat([summ_ews,  summ_drv,  summ_full],  ignore_index=True)\
          .to_csv(outdir / f"{args.var}_ablations_summary.csv", index=False)


    
if __name__ == "__main__":

    sn.set_style("white")
    plt.rc("figure", figsize=(13, 9))
    plt.rc("font", size=12)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    t0 = time.perf_counter()
    try:
        main()
    finally:
        dt = time.perf_counter() - t0
        print(f"\nTotal runtime: {dt/60:.2f} min")