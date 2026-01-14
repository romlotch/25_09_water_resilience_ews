#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from skimage.color import rgb2lab, lab2rgb

from utils.config import load_config, cfg_path

sn.set_style("white")
plt.rc("figure", figsize=(13, 9))
plt.rc("font", size=12)
plt.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


# -----------------------
# Color helpers
# -----------------------
HEX_LEFT  = "#924a4aff"  # reddish
HEX_RIGHT = "#49879bff"  # teal-ish
HEX_MID   = "#f7f7f7ff"  # neutral

def hex8_to_rgba_tuple(hex8: str):
    s = hex8.lstrip("#")
    r, g, b, a = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)
    return (r/255.0, g/255.0, b/255.0, a/255.0)

def rgb_to_lab_rows(rgb_rows: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(rgb_rows).astype(float)
    lab_img = rgb2lab(arr.reshape(-1, 1, 3))
    return lab_img.reshape(-1, 3)

def lab_to_rgb_rows(lab_rows: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(lab_rows).astype(float)
    rgb_img = lab2rgb(arr.reshape(-1, 1, 3))
    rgb = rgb_img.reshape(-1, 3)
    return np.clip(rgb, 0, 1)

def make_diverging_cmap(hex_left, hex_right, hex_mid="#f7f7f7ff", name="custom_div", N=256):
    c0 = np.array(hex8_to_rgba_tuple(hex_left))
    c1 = np.array(hex8_to_rgba_tuple(hex_mid))
    c2 = np.array(hex8_to_rgba_tuple(hex_right))

    n1 = N // 2 + 1
    n2 = N - n1 + 1
    t1 = np.linspace(0, 1, n1)[:, None]
    t2 = np.linspace(0, 1, n2)[:, None]

    lab0 = rgb_to_lab_rows(c0[:3])
    lab1 = rgb_to_lab_rows(c1[:3])
    lab2 = rgb_to_lab_rows(c2[:3])

    seg1_lab = lab0 + (lab1 - lab0) * t1
    seg2_lab = lab1 + (lab2 - lab1) * t2

    seg1_rgb = lab_to_rgb_rows(seg1_lab)
    seg2_rgb = lab_to_rgb_rows(seg2_lab)

    a0, a1, a2 = c0[3], c1[3], c2[3]
    seg1_a = (a0 + (a1 - a0) * t1.ravel())
    seg2_a = (a1 + (a2 - a1) * t2.ravel())

    rgb = np.vstack([seg1_rgb, seg2_rgb[1:]])
    a   = np.hstack([seg1_a, seg2_a[1:]])
    rgba = np.hstack([rgb, a[:, None]])

    return ListedColormap(rgba, name=name)




def open_and_align(delta_path: str, ts_path: str, kt_path: str, mc_path: str):
   
    ds_delta = xr.open_dataset(delta_path)
    ds_ts    = xr.open_dataset(ts_path)
    ds_kt    = xr.open_dataset(kt_path)
    ds_mc    = xr.open_dataset(mc_path)

    ds_delta, ds_ts, ds_kt, ds_mc = xr.align(ds_delta, ds_ts, ds_kt, ds_mc, join="inner")
    return ds_delta, ds_ts, ds_kt, ds_mc


def sign_to_class(x):
    return xr.where(x > 0, 1, xr.where(x < 0, -1, 0))


def classify_kt(ds_kt, var, suf, alpha):
    tau  = ds_kt[f"{var}_{suf}_kt"]
    pval = ds_kt[f"{var}_{suf}_pval"]
    sig = (pval < alpha)
    return sign_to_class(tau).where(sig, 0)


def classify_ts(ds_ts, var, suf, q):
    slope = ds_ts.get(f"{var}_{suf}_ts")
    if slope is None:
        ref = ds_ts[list(ds_ts.data_vars)[0]]
        return xr.full_like(ref, 0)
    sig_name = f"{var}_{suf}_ts_sig"
    if sig_name in ds_ts:
        sig = ds_ts[sig_name] > 0
    else:
        thr = float(np.nanquantile(np.abs(slope.values), q))
        sig = np.abs(slope) > thr
    return sign_to_class(slope).where(sig, 0)


def classify_delta(ds_delta, var, suf, q):
    base = ds_delta.get(f"{var}_delta_{suf}")
    if base is None:
        ref = ds_delta[list(ds_delta.data_vars)[0]]
        return xr.full_like(ref, 0)

    sig_name = f"{var}_delta_{suf}_sig"
    p_name   = f"{var}_delta_{suf}_pval"
    if sig_name in ds_delta:
        sig = ds_delta[sig_name].astype(bool)
    elif p_name in ds_delta:
        sig = ds_delta[p_name] < 0.05
    else:
        thr = float(np.nanquantile(np.abs(base.values), q))
        sig = np.abs(base) > thr
    return sign_to_class(base).where(sig, 0)


def classify_mc(ds_mc, var, suf, q):
    meanchange = ds_mc.get(f"{var}_{suf}_mean_change")
    if meanchange is None:
        ref = ds_mc[list(ds_mc.data_vars)[0]]
        return xr.full_like(ref, 0)

    sig_name = f"{var}_{suf}_mean_change_sig"
    p_name   = f"{var}_{suf}_ttest_p"
    if sig_name in ds_mc:
        sig = ds_mc[sig_name].astype(bool)
    elif p_name in ds_mc:
        sig = ds_mc[p_name] < 0.05
    else:
        thr = float(np.nanquantile(np.abs(meanchange.values), q))
        sig = np.abs(meanchange) > thr
    return sign_to_class(meanchange).where(sig, 0)


def _to_numpy(da):
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def confusion_3x3(a, b):
    av = _to_numpy(a).ravel()
    bv = _to_numpy(b).ravel()
    m = np.isfinite(av) & np.isfinite(bv)
    if not m.any():
        return np.zeros((3, 3), dtype=np.int64)
    av = av[m].astype(np.int8) + 1
    bv = bv[m].astype(np.int8) + 1
    idx = av * 3 + bv
    return np.bincount(idx, minlength=9).reshape(3, 3)


def kappa_weighted_linear(C):
    N = C.sum()
    if N == 0:
        return np.nan, np.nan
    W = np.array([[1.0, 0.5, 0.0],
                  [0.5, 1.0, 0.5],
                  [0.0, 0.5, 1.0]])
    Po = (W * C).sum() / N
    r = C.sum(axis=1)
    c = C.sum(axis=0)
    Exp = np.outer(r, c) / N
    Pe = (W * Exp).sum() / N
    if Pe == 1:
        return np.nan, Po
    return (Po - Pe) / (1 - Pe), Po


def percent_exact(C):
    N = C.sum()
    return np.nan if N == 0 else C.trace() / N


def kappa_on_union_nonneutral(a, b):
    m = ((a != 0) | (b != 0))
    A = a.where(m)
    B = b.where(m)
    C = confusion_3x3(A, B)
    kappa_w, _ = kappa_weighted_linear(C)
    return kappa_w


def jaccard_dir(a, b, sign=+1):
    A = _to_numpy((a == sign))
    B = _to_numpy((b == sign))
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return np.nan if union == 0 else inter / union


def compute_agreement(delta_path, ts_path, kt_path, mc_path,
                      var, indicators,
                      alpha_kt, delta_q, ts_q, mc_q):
    ds_delta, ds_ts, ds_kt, ds_mc = open_and_align(delta_path, ts_path, kt_path, mc_path)

    classes = {}
    for suf, _ in indicators:
        classes[("KT", suf)] = classify_kt(ds_kt, var, suf, alpha=alpha_kt)
        classes[("TS", suf)] = classify_ts(ds_ts, var, suf, q=ts_q)
        classes[("Δ",  suf)] = classify_delta(ds_delta, var, suf, q=delta_q)
        classes[("MC", suf)] = classify_mc(ds_mc, var, suf, q=mc_q)

    for k in classes:
        classes[k] = classes[k].transpose("lat", "lon")

    pairs = [("KT","TS"), ("KT","Δ"), ("KT","MC"), ("TS","Δ"), ("TS","MC"), ("Δ","MC")]

    rows = []
    for suf, label in indicators:
        for m1, m2 in pairs:
            C = confusion_3x3(classes[(m1, suf)], classes[(m2, suf)])
            kappa_w, _ = kappa_weighted_linear(C)
            acc = percent_exact(C)
            rows.append({
                "indicator": label,
                "pair": f"{m1} vs {m2}",
                "kappa_w": kappa_w,
                "percent_exact": acc,
                "n": int(C.sum())
            })
    df_agree = pd.DataFrame(rows)
    df_agree_pivot = df_agree.pivot(index="indicator", columns="pair", values="kappa_w")

    rows = []
    for suf, label in indicators:
        for m1, m2 in pairs:
            a, b = classes[(m1, suf)], classes[(m2, suf)]
            rows.append({
                "indicator": label,
                "pair": f"{m1} vs {m2}",
                "kappa_nonneutral": kappa_on_union_nonneutral(a, b),
                "Jaccard_inc": jaccard_dir(a, b, +1),
                "Jaccard_dec": jaccard_dir(a, b, -1),
            })
    df_aux = pd.DataFrame(rows)

    return df_aux, df_agree, df_agree_pivot


def plot_agreement(df_agree, df_agree_pivot, cmap, fig_dir: Path, var: str):
    desired = ["AC1", "SD", "FD", "Skew.", "Skew", "Kurt"]
    index_order = [lab for lab in desired if lab in df_agree_pivot.index]
    index_order += [lab for lab in df_agree_pivot.index if lab not in index_order]
    df_agree_pivot = df_agree_pivot.reindex(index=index_order)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    hm = sn.heatmap(
        df_agree_pivot,
        vmin=0, vmax=1,
        cmap=cmap,
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Weighted κ [linear]"},
        ax=ax,
        annot=False,
        square=True
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=12)

    mappable = hm.collections[0]
    cmap_obj = mappable.cmap
    norm = mappable.norm

    def luminance(rgb):
        r, g, b = rgb[:3]
        return 0.2126*r + 0.7152*g + 0.0722*b

    for i, ind in enumerate(df_agree_pivot.index):
        for j, pair in enumerate(df_agree_pivot.columns):
            val = df_agree_pivot.loc[ind, pair]
            if np.isfinite(val):
                rgba = cmap_obj(norm(val))
                text_color = "white" if luminance(rgba) < 0.5 else "black"
                txt = f"{val:0.2f}"
            else:
                text_color = "gray"
                txt = "–"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=12, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{var}_agreement_heatmap.svg"
    plt.tight_layout()
    plt.savefig(out, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def run(var: str, cfg, outdir=None, delta_path=None, ts_path=None, kt_path=None, mc_path=None):
    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))
    zarr_dir = outputs_root / "zarr"

    # default inferred inputs
    delta_path = str(delta_path) if delta_path else str(zarr_dir / f"out_{var}.zarr")
    ts_path    = str(ts_path)    if ts_path    else str(zarr_dir / f"out_{var}_ts.zarr")
    kt_path    = str(kt_path)    if kt_path    else str(zarr_dir / f"out_{var}_kt.zarr")
    mc_path    = str(mc_path)    if mc_path    else str(zarr_dir / f"out_{var}_meanchange.zarr")

    fig_dir = Path(outdir) if outdir else (outputs_root / "figures" / "agreement")

    alpha_kt = 0.05
    delta_q = 0.975
    ts_q = 0.975
    mc_q = 0.975
    indicators = [("ac1","AC1"), ("std","SD"), ("skew","Skew"), ("kurt","Kurt"), ("fd","FD")]

    # your teal sequential palette
    teal = "#49879bff"
    cmap_seq = sn.light_palette(teal, as_cmap=True)

    df_aux, df_agree, df_agree_pivot = compute_agreement(
        delta_path, ts_path, kt_path, mc_path,
        var=var,
        indicators=indicators,
        alpha_kt=alpha_kt,
        delta_q=delta_q,
        ts_q=ts_q,
        mc_q=mc_q,
    )

    # save tables 
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_aux.to_csv(fig_dir / f"{var}_agreement_aux_metrics.csv", index=False)
    df_agree.to_csv(fig_dir / f"{var}_agreement_kappa_table.csv", index=False)

    plot_agreement(df_agree, df_agree_pivot, cmap_seq, fig_dir=fig_dir, var=var)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--var", required=True, choices=["sm", "Et", "precip"], help="Variable name")
    p.add_argument("--outdir", default=None, help="Optional override output directory")

    # Optional path overrides 
    p.add_argument("--delta_path", default=None)
    p.add_argument("--ts_path", default=None)
    p.add_argument("--kt_path", default=None)
    p.add_argument("--mc_path", default=None)

    args = p.parse_args()
    cfg = load_config(args.config)

    run(
        var=args.var,
        cfg=cfg,
        outdir=args.outdir,
        delta_path=args.delta_path,
        ts_path=args.ts_path,
        kt_path=args.kt_path,
        mc_path=args.mc_path,
    )