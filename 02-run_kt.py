#!/usr/bin/env python3
import os
import sys
import argparse
import builtins
import psutil._common
import numpy as np
import xarray as xr
from scipy.stats import kendalltau
from concurrent.futures import ProcessPoolExecutor, as_completed



'''
Executable function to run kendall tau on a set of already computed
EWS time series.

If you pass a variable name, it calculates the kendall tau just for the variable (e.g. precip),
instead of calculating based on the suffixes of the EWS 

## Run on variable EWS
python 02-run_kt.py  --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr --workers 8
python 02-run_kt.py  --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_et.zarr --workers 4

## Run on masked breakpoint EWS pos 
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc.zarr --workers 4

## RUn on masked breakpoints EWS neg 
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc_neg.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc_neg.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc_neg.zarr --workers 4



## Run on aridity
python 250328_kendall_tau_new.py /mnt/data/romi/data/et_pot/aridity_index.zarr

'''


def ktfunc(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute kendall‐τ & p‐value for a single DataArray."""
    def _kt(y):
        if not np.all(np.isnan(y)):
            x = np.arange(len(y)) / 12
            kt, p = kendalltau(x, y, nan_policy="omit")
            return kt, p
        return np.nan, np.nan

    tau, p = xr.apply_ufunc(
        _kt,
        ds[var],
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64", "float64"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    out = tau.to_dataset(name=f"{var}_kt")
    out[f"{var}_pval"] = p
    return out


def _maybe_apply_precip_landmask(ds: xr.Dataset, varname: str) -> xr.Dataset:
    """
    If varname is 'precip' or starts with 'precip_', apply land/sea mask
    before anything else. Expects ds to have coords (lat, lon) but mask has latitude longitde.
    """
    if not (varname == "precip" or varname.startswith("precip_")):
        return ds

    # Open land/sea mask 
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
    return ds.where(land)


def _worker_compute(varname, input_path, tmpl_store, tmp_dir):
    """
    For suffix 'varname', open input_path, compute kendall-τ,
    and write to a temporary store under tmp_dir/varname.zarr
    Returns the path to the temp store.
    """
    ds = xr.open_zarr(input_path).chunk({"time": -1, "lat": 50, "lon": 50})

    ds = _maybe_apply_precip_landmask(ds, varname)

    tau_ds = ktfunc(ds, varname)
    
    for v in tau_ds.data_vars:
        tau_ds[v].encoding.pop("chunks", None)
    
    tau_ds = tau_ds.chunk({"lat": 50, "lon": 50})
    
    out_store = os.path.join(tmp_dir, f"{varname}.zarr")
    if os.path.exists(out_store):
        # in case of restart
        import shutil
        shutil.rmtree(out_store)
    tau_ds.to_zarr(out_store, mode="w")
    return out_store


def main():
    p = argparse.ArgumentParser(
        description="Compute Kendall tau in parallel, then merge."
    )
    p.add_argument("--input",   required=True, help="path/to/input.zarr")
    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        print("Input not found:", inp, file=sys.stderr)
        sys.exit(1)

    # Figure out which vars to do
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
    final_store = os.path.join(os.path.dirname(inp), base + "_kt.zarr")
    if os.path.exists(final_store):
        print("Output already exists:", final_store)
        sys.exit(0)

    # Make a temp dir for per-var stores
    tmp_dir = os.path.join(os.path.dirname(inp), base + "_kt_temp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Launch per-var workers
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(_worker_compute, v, inp, final_store, tmp_dir): v
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
    
    # Open each temp, rename its variables into root
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