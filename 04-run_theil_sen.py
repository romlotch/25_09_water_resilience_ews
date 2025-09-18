#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import xarray as xr
from scipy.stats import theilslopes
from concurrent.futures import ProcessPoolExecutor, as_completed




'''
Executable function to run Theil–Sen slope estimation on a set of already computed
EWS time series.

Robust Theil–Sen slope + confidence interval. Significance is flagged when the CI
excludes zero.

Assumptions:
- Time spacing is weekly; we convert to "years" using x = arange(n)/12 to get
  slope units per year. If your data isn’t monthly, adjust the divisor.

Example runs:
python 04-run_theil_sen.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr --workers 8
python 04-run_theil_sen.py --input /mnt/data/romi/output/paper_1/output_persiann_cdr/out_persiann_cdr_weekly_masked.zarr --workers 8
python 04-run_thiel_sen.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr --workers 8

# Masked breakpoint EWS
python 250328_theil_sen_slope.py --input /mnt/data/romi/output/paper_1/output_changepoints/soil_moisture/sm_ews_breakpoints_pettitt.zarr --workers 8
python 250328_theil_sen_slope.py --input /mnt/data/romi/output/paper_1/output_changepoints/transpiration/Et_ews_breakpoints_pettitt.zarr --workers 8
'''


def tsfunc(ds: xr.Dataset, var: str, alpha: float = 0.05) -> xr.Dataset:
    """Compute Theil–Sen slope (+ CI) along 'time' for a single DataArray.

    Outputs:
      var_ts     : slope per year
      var_ts_lo  : lower CI bound
      var_ts_hi  : upper CI bound
      var_ts_sig : 1 if CI excludes 0, else 0
    """

    # SciPy's theilslopes returns (slope, intercept, lo_slope, up_slope) at (1-alpha) CI.
    # Data is weekly, convert x to years to get slope per year.
    def _ts(y):
        # y is a 1D numpy array along time for one (lat, lon) cell
        if y.size == 0 or np.all(np.isnan(y)):
            return np.nan, np.nan, np.nan, np.nan

        x = np.arange(y.size) / 52.0  # years 

        # Pairwise drop NaNs (keep only positions where y is finite)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan, np.nan, np.nan, np.nan

        xv = x[mask]
        yv = y[mask]

        # Need at least two distinct x values
        if np.unique(xv).size < 2:
            return np.nan, np.nan, np.nan, np.nan

        try:
            slope, intercept, lo, hi = theilslopes(yv, xv, alpha=alpha)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

        # significance: CI excludes 0
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
    For suffix 'varname', open input_path, compute Theil–Sen slope set,
    and write to a temporary store under tmp_dir/varname.zarr
    Returns the path to the temp store.
    """
    ds = xr.open_zarr(input_path).chunk({"time": -1, "lat": 50, "lon": 50})
    ts_ds = tsfunc(ds, varname, alpha=alpha)

    # Ensure clean encodings for zarr
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
    p = argparse.ArgumentParser(
        description="Compute Theil–Sen slope in parallel, then merge."
    )
    p.add_argument("--input",   required=True, help="path/to/input.zarr")
    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    p.add_argument("--alpha",   type=float, default=0.05,
                   help="Significance level for CI (default 0.05: 95% CI)")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        print("Input not found:", inp, file=sys.stderr)
        sys.exit(1)

    # Choose variables (same suffix logic as your original script)
    ds0 = xr.open_zarr(inp, chunks={})
    suffixes = ("ac1", "std", "skew", "kurt", "fd")
    vars_ = [
        v for v in ds0.data_vars
        if any(v.endswith(f"_{s}") for s in suffixes)
           and "time" in ds0[v].dims
           and v.count("_")==1
    ]
    ds0.close()
    if not vars_:
        print("No matching variables found.", file=sys.stderr)
        sys.exit(1)
    print("Will process:", vars_)

    # Prepare final output path
    base = os.path.splitext(os.path.basename(inp))[0]
    final_store = os.path.join(os.path.dirname(inp), base + "_ts.zarr")
    if os.path.exists(final_store):
        print("Output already exists:", final_store)
        sys.exit(0)

    # Make a temp dir for per-var stores
    tmp_dir = os.path.join(os.path.dirname(inp), base + "_ts_temp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Launch per-var workers
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(_worker_compute, v, inp, tmp_dir, args.alpha): v
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

    # Merge all the per-var stores into one
    print("Merging into final store:", final_store)
    parts = [xr.open_zarr(os.path.join(tmp_dir, f"{v}.zarr")) for v in vars_]
    merged = xr.merge(parts)

    # Write merged
    merged.chunk({"lat":50,"lon":50}).to_zarr(final_store, mode="w", consolidated=True)
    print("All done. Final store at:", final_store)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()