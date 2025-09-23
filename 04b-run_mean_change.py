import os
import sys
import argparse
import numpy as np
import xarray as xr
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
Compute per-pixel change in mean between first and second half
of each indicator time series, plus Welch's t-test (independent samples,
unequal variances) and a significance flag.

- Split the series at mid = len(time)//2, then compare equal-length halves.
- nan_policy='omit'

Eg.:
    python 04b-run_mean_change.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr --workers 4
    python 04b-run_mean_change.py --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr --workers 2
    python 04b-run_mean_change.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr --workers 4
"""


def mean_change_func(ds, var, alpha = 0.05):

    """Compute change in mean (second-half minus first-half) along time plus Welch's t-test."""

    # Determine equal-length halves based on time 
    n_time = ds.dims.get("time", None)
    if n_time is None or n_time < 2:
        # Not enough data to split sensibly
        da = ds[var]
        out = xr.full_like(da.isel(time=0, drop=True), fill_value=np.nan).to_dataset(name=f"{var}_mean_change")
        out[f"{var}_ttest_t"] = xr.full_like(out[f"{var}_mean_change"], np.nan)
        out[f"{var}_ttest_p"] = xr.full_like(out[f"{var}_mean_change"], np.nan)
        out[f"{var}_mean_change_sig"] = xr.full_like(out[f"{var}_mean_change"], np.nan)
        return out

    mid = n_time // 2
    len_half = min(mid, n_time - mid)  # equalize lengths if odd 

    # Slice equal-length halves
    first_half = ds[var].isel(time=slice(mid - len_half, mid))
    second_half = ds[var].isel(time=slice(mid, mid + len_half))

    # Reindex time to 0..len_half-1 so apply_ufunc has aligned dimensions (fails otherwise)
    first_half = first_half.assign_coords(time=np.arange(len_half))
    second_half = second_half.assign_coords(time=np.arange(len_half))

    def _diff_and_t(a, b):
        mean_a = np.nanmean(a) if np.any(np.isfinite(a)) else np.nan
        mean_b = np.nanmean(b) if np.any(np.isfinite(b)) else np.nan
        diff = mean_b - mean_a

        # If fewer than 2 total points per group,return nan
        n_a = np.isfinite(a).sum()
        n_b = np.isfinite(b).sum()
        if n_a < 2 or n_b < 2:
            return diff, np.nan, np.nan, np.nan

        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        sig = float(p < alpha) if np.isfinite(p) else np.nan
        return diff, t, p, sig

    diff, tstat, pval, sig = xr.apply_ufunc(
        _diff_and_t,
        first_half,
        second_half,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64", "float64", "float64", "float64"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    out = diff.to_dataset(name=f"{var}_mean_change")
    out[f"{var}_ttest_t"] = tstat
    out[f"{var}_ttest_p"] = pval
    out[f"{var}_mean_change_sig"] = sig
    return out


def _worker_compute(varname, input_path, tmp_dir, alpha):
    """
    For var, open input_path, compute mean change and t-test,
    and write to a temporary store under tmp_dir/varname.zarr
    Returns the path to the temp store.
    """
    ds = xr.open_zarr(input_path).chunk({"time": -1, "lat": 50, "lon": 50})
    out_ds = mean_change_func(ds, varname, alpha=alpha)

    # Clean encoding in case it makes it faster
    for v in out_ds.data_vars:
        out_ds[v].encoding.pop("chunks", None)

    out_ds = out_ds.chunk({"lat": 50, "lon": 50})

    out_store = os.path.join(tmp_dir, f"{varname}.zarr")
    if os.path.exists(out_store):
        import shutil
        shutil.rmtree(out_store)
    out_ds.to_zarr(out_store, mode="w")
    return out_store


def main():
    p = argparse.ArgumentParser(
        description="Compute change in mean (second half minus first) with Welch's t-test in parallel, then merge."
    )
    p.add_argument("--input",   required=True, help="path/to/input.zarr")
    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    p.add_argument("--alpha",   type=float, default=0.05,
                   help="Significance threshold for p-value (default 0.05)")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        print("Input not found:", inp, file=sys.stderr)
        sys.exit(1)

    # Choose variables
    ds0 = xr.open_zarr(inp, chunks={})
    suffixes = ("ac1", "std", "skew", "kurt", "fd")
    vars_ = [
        v for v in ds0.data_vars
        if any(v.endswith(f"_{s}") for s in suffixes)
           and "time" in ds0[v].dims
           and v.count("_") == 1
    ]
    ds0.close()
    if not vars_:
        print("No matching variables found!", file=sys.stderr)
        sys.exit(1)
    print("Will process:", vars_)

    # Prep final out path
    base = os.path.splitext(os.path.basename(inp))[0]
    final_store = os.path.join(os.path.dirname(inp), base + "_meanchange.zarr")
    if os.path.exists(final_store):
        print("Output already exists:", final_store)
        sys.exit(0)

    # Temp dir 
    tmp_dir = os.path.join(os.path.dirname(inp), base + "_meanchange_temp")
    os.makedirs(tmp_dir, exist_ok=True)

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

    # Merge all per-var stores into one
    print("Merging into final store:", final_store)
    parts = [xr.open_zarr(os.path.join(tmp_dir, f"{v}.zarr")) for v in vars_]
    merged = xr.merge(parts)

    # Write merged
    merged.chunk({"lat": 50, "lon": 50}).to_zarr(final_store, mode="w", consolidated=True)
    print("All done. Final store at:", final_store)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()