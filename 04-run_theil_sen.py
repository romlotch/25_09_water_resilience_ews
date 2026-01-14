#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import xarray as xr
from scipy.stats import theilslopes
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.config import load_config, cfg_path, cfg_get
from pathlib import Path


"""
Run Theil–Sen slope estimation on a set of already computed
EWS time series.

Robust Theil–Sen slope + confidence interval. Significance iif the CI
excludes zero.

E.g.:
python 04-run_theil_sen.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr --workers 8
python 04-run_theil_sen.py --input /mnt/data/romi/output/paper_1/output_persiann_cdr/out_persiann_cdr_weekly_masked.zarr --workers 8
python 04-run_thiel_sen.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr --workers 8

# Masked breakpoint EWS
python 250328_theil_sen_slope.py --input /mnt/data/romi/output/paper_1/output_changepoints/soil_moisture/sm_ews_breakpoints_pettitt.zarr --workers 8
python 250328_theil_sen_slope.py --input /mnt/data/romi/output/paper_1/output_changepoints/transpiration/Et_ews_breakpoints_pettitt.zarr --workers 8
"""

def tsfunc(ds, var, alpha=0.05):
    # Assumes data is weekly, convert x to years to get slope per year.
    def _ts(y):
        if y.size == 0 or np.all(np.isnan(y)):
            return np.nan, np.nan, np.nan, np.nan

        x = np.arange(y.size) / 52.0  # years

        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan, np.nan, np.nan, np.nan

        xv = x[mask]
        yv = y[mask]

        if np.unique(xv).size < 2:
            return np.nan, np.nan, np.nan, np.nan

        try:
            slope, intercept, lo, hi = theilslopes(yv, xv, alpha=alpha)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

        sig = float((lo > 0) or (hi < 0))
        return slope, lo, hi, sig

    slope, lo, hi, sig = xr.apply_ufunc(
        _ts,
        ds[var],
        input_core_dims=[["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64", "float64", "float64", "float64"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    out = slope.to_dataset(name=f"{var}_ts")
    out[f"{var}_ts_lo"] = lo
    out[f"{var}_ts_hi"] = hi
    out[f"{var}_ts_sig"] = sig
    return out


def _worker_compute(varname, input_path, tmp_dir, alpha):
    """
    For varname, open input_path, compute Theil–Sen slope set,
    and write to a temporary store under tmp_dir/varname.zarr
    Returns the path to the temp store.
    """
    ds = xr.open_dataset(input_path).chunk({"time": -1, "lat": 50, "lon": 50})
    ts_ds = tsfunc(ds, varname, alpha=alpha)

    for v in ts_ds.data_vars:
        ts_ds[v].encoding.pop("chunks", None)

    ts_ds = ts_ds.chunk({"lat": 50, "lon": 50})

    out_store = os.path.join(tmp_dir, f"{varname}.zarr")
    if os.path.exists(out_store):
        import shutil
        shutil.rmtree(out_store)

    ts_ds.to_zarr(out_store, mode="w")
    return out_store


def main():
    p = argparse.ArgumentParser(description="Compute Theil–Sen in parallel then merge.")
    p.add_argument("--var", default=None,
                   help="Optional: used only to infer --input if --input is omitted (e.g., sm, Et, precip).")
    p.add_argument("--suffix", default=None,
                   help="Optional suffix for inferred input (e.g. breakpoint_pettitt).")
    p.add_argument("--input", default=None,
                   help="Optional override path to input zarr store. If omitted, inferred from config + --var + --suffix.")
    p.add_argument("--output", default=None,
                   help="Optional override path for output *_ts.zarr. If omitted, written next to input as <input>_ts.zarr.")

    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for CI (default 0.05: 95% CI)")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = Path(cfg_path(cfg, "paths.outputs_root", must_exist=True))

    def _sfx(s):
        if not s:
            return ""
        s = str(s).strip()
        return s if s.startswith("_") else f"_{s}"

    if args.input:
        inp = Path(args.input)
    else:
        if not args.var:
            print("If --input is omitted, you must provide --var (e.g., sm, Et, precip).", file=sys.stderr)
            sys.exit(1)
        inp = outputs_root / "zarr" / f"out_{args.var}{_sfx(args.suffix)}.zarr"

    if not inp.exists():
        print("Input not found:", str(inp), file=sys.stderr)
        sys.exit(1)

    ds0 = xr.open_dataset(str(inp), chunks={})
    suffixes = ("ac1", "std", "skew", "kurt", "fd")
    vars_ = [
        v for v in ds0.data_vars
        if any(v.endswith(f"_{s}") for s in suffixes)
        and "time" in ds0[v].dims
        and v.count("_") == 1
    ]
    ds0.close()

    if not vars_:
        print("No matching variables found.", file=sys.stderr)
        sys.exit(1)
    print("Will process:", vars_)

    base = inp.stem
    final_store = Path(args.output) if args.output else inp.with_name(base + "_ts.zarr")
    if final_store.exists():
        print("Output already exists:", str(final_store))
        sys.exit(0)

    tmp_dir = inp.with_name(base + "_ts_temp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(_worker_compute, v, str(inp), str(tmp_dir), args.alpha): v
            for v in vars_
        }
        done = 0
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                store_path = fut.result()
                done += 1
                print(f"[{done}/{len(vars_)}] done {v} → {store_path}")
            except Exception as e:
                print(f"[ERROR] {v}: {e}", file=sys.stderr)

    print("Merging into final store:", str(final_store))
    parts = [xr.open_dataset(str(tmp_dir / f"{v}.zarr")) for v in vars_]
    merged = xr.merge(parts)

    merged.chunk({"lat": 50, "lon": 50}).to_zarr(str(final_store), mode="w", consolidated=True)
    print("All done. Final store at:", str(final_store))

    import shutil
    shutil.rmtree(str(tmp_dir))


if __name__ == "__main__":
    main()