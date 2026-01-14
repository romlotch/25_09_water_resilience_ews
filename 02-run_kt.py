#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
from scipy.stats import kendalltau
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dask.distributed import Client, LocalCluster
from utils.config import load_config, cfg_path
from pathlib import Path


'''
Function to run kendall tau on a set of already computed
EWS time series.

If you pass a variable name, it calculates the kendall tau just for the variable (e.g. precip),
instead of calculating based on the suffixes of the EWS 

Path to land sea mask is hardcoded. 

## Run on variable EWS
python 02-run_kt.py  --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr --workers 8
python 02-run_kt.py  --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_et.zarr --workers 4

## Run on masked breakpoint EWS pos 
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc.zarr --workers 4

## Run on masked breakpoints EWS neg 
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_sm_final/out_sm_breakpoint_stc_neg.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_Et_final/out_Et_breakpoint_stc_neg.zarr --workers 8
python 02-run_kt.py --input /mnt/data/romi/output/paper_1/output_precip_final/out_precip_breakpoint_stc_neg.zarr --workers 4

## Run on aridity
python 250328_kendall_tau_new.py /mnt/data/romi/data/et_pot/aridity_index.zarr


04.09.25 

Updated for better parallelisation. Previously had one job per variable, now includes dask parallelisation. 
Also adds an optional trimming CLI option to trim the last 2.5 years 

    E.g. positives
    python 02-run_kt.py --input /home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc.zarr --dask-workers 16 --trim
    python 02-run_kt.py --input /home/romi/ews/output/output_sm_final/out_sm_breakpoint_stc.zarr --dask-workers 16 --trim
    python 02-run_kt.py --input /home/romi/ews/output/output_precip_final/out_precip_breakpoint_stc.zarr --dask-workers 16 --trim

    E.g. negatives
    python 02-run_kt.py --input /home/romi/ews/output/output_Et_final/out_Et_breakpoint_stc_neg.zarr --dask-workers 16 --trim
    python 02-run_kt.py --input /home/romi/ews/output/output_sm_final/out_sm_breakpoint_stc_neg.zarr --dask-workers 16 --trim
    python 02-run_kt.py --input /home/romi/ews/output/output_precip_final/out_precip_breakpoint_stc_neg.zarr --dask-workers 16 --trim


'''

def _trim(ds: xr.Dataset, years: int = 2) -> xr.Dataset:
    """
    If trimming would remove everything (or time axis is empty), returns ds unchanged.
    """
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        return ds

    tmax = pd.to_datetime(ds["time"].values[-1])
    cut  = np.datetime64(tmax - pd.DateOffset(months=30))  # 2.5 years

    tmin = ds["time"].values[0]
    if cut <= tmin:
        return ds

    return ds.sel(time=slice(None, cut))


def ktfunc(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Compute kendall‐tau & p‐value for a single DataArray."""
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


def _maybe_apply_precip_landmask(ds: xr.Dataset, varname: str, cfg) -> xr.Dataset:
    """
    If varname is 'precip' or starts with 'precip_', apply land/sea mask
    before anything else. Expects ds to have coords (lat, lon) but mask has latitude longitde.
    """
    if not (varname == "precip" or varname.startswith("precip_")):
        return ds

    # Open land/sea mask 
    mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
    mask_ds = xr.open_dataset(mask_path, engine="cfgrib")

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


def _worker_compute(varname, input_path, tmpl_store, tmp_dir, trim=False, cfg=None) -> str:
    """
    For suffix 'varname', open input_path, compute kendall-τ,
    and write to a temporary store under tmp_dir/varname.zarr
    Returns the path to the temp store.
    """
    ds = xr.open_zarr(input_path).chunk({"time": -1, "lat": 100, "lon": 100})

    ds = _maybe_apply_precip_landmask(ds, varname, cfg)
    if trim:
        ds = _trim(ds)

    tau_ds = ktfunc(ds, varname)
    
    for v in tau_ds.data_vars:
        tau_ds[v].encoding.pop("chunks", None)
    
    tau_ds = tau_ds.chunk({"lat": 100, "lon": 100})
    
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
    p.add_argument("--variable", required=False, default=None,
               help="Variable key (sm|Et|precip). Used to infer --input if not provided.")
    p.add_argument("--suffix", required=False, default=None,
                help="Optional suffix (e.g. breakpoint_stc). Used to infer --input if not provided.")
    p.add_argument("--input", required=False, default=None,
                help="Optional override input zarr. If omitted, inferred from config + --variable + --suffix.")
    p.add_argument("--workers", type=int, default=4, help="Processes if not using Dask")
    p.add_argument("--dask-workers", type=int, default=0, help="Use Dask LocalCluster with this many workers (threads_per_worker=1)")
    p.add_argument("--trim", action="store_true")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)

    def _sfx(s):
        if not s: return ""
        return s if str(s).startswith("_") else f"_{s}"

    if args.input is None:
        if args.variable is None:
            raise ValueError("Provide --input or --variable (and optionally --suffix).")
        input_path = Path(outputs_root) / "zarr" / f"out_{args.variable}{_sfx(args.suffix)}.zarr"
    else:
        input_path = Path(args.input)


    maybe_client_ctx = nullcontext()

    if args.dask_workers > 0:
        cluster = LocalCluster(n_workers=args.dask_workers, threads_per_worker=1, processes=True)
        maybe_client_ctx = Client(cluster)

    with maybe_client_ctx:

        # Figure out which vars to do
        ds0 = xr.open_zarr(str(input_path), chunks={})
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
        
        base = os.path.splitext(os.path.basename(str(input_path)))[0]
        final_store = os.path.join(os.path.dirname(str(input_path)), base + "_kt.zarr")
        if os.path.exists(final_store):
            print("Output already exists:", final_store); sys.exit(0)

        base = input_path.stem  # e.g. out_sm or out_sm_breakpoint_stc
        zarr_dir = Path(outputs_root) / "zarr"
        zarr_dir.mkdir(parents=True, exist_ok=True)

        final_store = zarr_dir / f"{base}_kt.zarr"
        tmp_dir = zarr_dir / f"{base}_kt_temp"

        if args.dask_workers > 0: 

            # if dask workers is specified, process the vars squentially and distribute across cluster 
            for v in vars_:
                ds = xr.open_zarr(str(input_path)).chunk({"time": -1, "lat": 100, "lon": 100})
                ds = _maybe_apply_precip_landmask(ds, v, cfg)
                if args.trim:
                    ds = _trim(ds)
                tau_ds = ktfunc(ds, v)  # returns Dask arrays
                tau_ds = tau_ds.chunk({"lat": 100, "lon": 100})
                tau_ds.to_zarr(os.path.join(str(tmp_dir), f"{v}.zarr"), mode="w", consolidated=True)

        else: 

            # if not just use process pools 
            with ProcessPoolExecutor(max_workers=args.workers) as exe:
                futures = {
                            exe.submit(_worker_compute, v, str(input_path), str(final_store), str(tmp_dir), args.trim, cfg): v
                            for v in vars_
                        }
                done = 0
                for fut in as_completed(futures):
                    v = futures[fut]
                    try:
                        fut.result(); done += 1
                        print(f"[{done}/{len(vars_)}] done {v}")
                    except Exception as e:
                        print(f"[ERROR] {v}: {e}", file=sys.stderr)


        # merge per-variable outputs
        parts = [xr.open_zarr(os.path.join(str(tmp_dir), f"{v}.zarr")) for v in vars_]
        merged = xr.merge(parts)
        merged.chunk({"lat": 100, "lon": 100}).to_zarr(str(final_store), mode="w", consolidated=True)
        for pds in parts:
            pds.close()
        print("All done. Final store at:", str(final_store))

        # cleanup
        import shutil
        shutil.rmtree(str(tmp_dir))

    
if __name__ == "__main__":
    main()