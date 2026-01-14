import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from pathlib import Path
from utils.config import load_config, cfg_path, cfg_get

""" 
After running the EWS on the variable, and the changepoint script on the variable, 
use this script to mask the EWS to include only those with a breakpoint that
occured late enough in the time series (12 years; approx. half the time series plus 2 
years to account for the rolling window length) to re-run the EWS on the period before
the changepoint, and only if they were significant. 

If the var_name is precip, it will land-mask the dataset. Filepath to land mask
file is currently hard-coded. 

Saves four output files, one for each changepoint test, and one combined. 

The EWS script is then re-run on these output files. 

E.g.
    python 05-mask_breakpoints.py --ews_ds_path '/mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr' --var 'sm' --out_dir '/mnt/data/romi/output/paper_1/output_sm_final'
    python 05-mask_breakpoints.py --ews_ds_path '/mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr' --var 'Et' --out_dir '/mnt/data/romi/output/paper_1/output_Et_final'
    python 05-mask_breakpoints.py --ews_ds_path '/mnt/data/romi/output/paper_1/output_precip_final/out_precip.zarr' --var 'precip' --out_dir '/mnt/data/romi/output/paper_1/output_precip_final'
"""


def infer_cp_path(ews_path: str) -> str:
    """
    Insert '_chp' before the file/dir suffix:
      /.../out_sm.zarr        -> /.../out_sm_chp.zarr
      /.../out_precip.zarr    -> /.../out_precip_chp.zarr
    """
    p = Path(ews_path)
    if p.suffixes:
        cp_name = p.stem + "_chp" + "".join(p.suffixes)
    else:
        cp_name = p.name + "_chp"
    return str(p.with_name(cp_name))


def mask_breakpoints(ews_ds, var_name, ds_breaks, output_dir):
    alpha = 0.05

    # mask significant pixels
    pettitt_sig = (ds_breaks["pettitt_pval"] < alpha) & (ds_breaks["pettitt_cp"] > 0)
    stc_sig     = (ds_breaks["Fstat_pval"]   < alpha) & (ds_breaks["strucchange_bp"] > 0)
    var_sig     = (ds_breaks["pval_var"]     < alpha) & (ds_breaks["bp_var"] > 0)

    def break_times(cp_idx_da):
        """
        Extract time of breakpoint from index array.
        """
        cp = cp_idx_da.values.astype(int)
        times = ews_ds["time"].values

        out = np.full(cp.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = (cp >= 0) & (cp < times.size)
        out[valid] = times[cp[valid]]

        return xr.DataArray(
            out,
            dims=cp_idx_da.dims,
            coords=cp_idx_da.coords,
            name=f"{cp_idx_da.name}_time",
        )

    time = ews_ds["time"]

    # threshold: 12 years after series start
    t0 = pd.to_datetime(ews_ds["time"].values[0])
    t_thresh = np.datetime64(t0 + pd.DateOffset(years=12))

    # breakpoint times
    bp_time_pettitt = break_times(ds_breaks["pettitt_cp"])
    bp_time_stc     = break_times(ds_breaks["strucchange_bp"])
    bp_time_var     = break_times(ds_breaks["bp_var"])

    # only late-enough + significant
    pettitt_late = pettitt_sig & (bp_time_pettitt >= t_thresh)
    stc_late     = stc_sig     & (bp_time_stc     >= t_thresh)
    var_late     = var_sig     & (bp_time_var     >= t_thresh)
    combined_mask = pettitt_late | stc_late | var_late

    # mask out time after breakpoint
    sm_ds = ews_ds[[f"{var_name}"]]

    sm_p = sm_ds.where(pettitt_late)
    time_mask_p = time <= bp_time_pettitt
    ds_masked_pettitt = sm_p.where(time_mask_p)

    sm_s = sm_ds.where(stc_late)
    time_mask_s = time <= bp_time_stc
    ds_masked_stc = sm_s.where(time_mask_s)

    sm_v = sm_ds.where(var_late)
    time_mask_v = time <= bp_time_var
    ds_masked_var = sm_v.where(time_mask_v)

    sm_c = sm_ds.where(combined_mask)
    bp_time_any = xr.concat([bp_time_pettitt, bp_time_stc, bp_time_var], dim="band").min(dim="band")
    time_mask_c = time <= bp_time_any
    ds_masked_all = sm_c.where(time_mask_c)

    # save
    output_dir = Path(output_dir)
    ds_masked_all.to_zarr(str(output_dir / f"{var_name}_cp_masked_all.zarr"), mode="w")
    ds_masked_pettitt.to_zarr(str(output_dir / f"{var_name}_cp_masked_pettitt.zarr"), mode="w")
    ds_masked_stc.to_zarr(str(output_dir / f"{var_name}_cp_masked_stc.zarr"), mode="w")
    ds_masked_var.to_zarr(str(output_dir / f"{var_name}_cp_masked_var.zarr"), mode="w")


def mask_negatives(ews_ds, var_name, ds_breaks, output_dir,
                   alpha=0.05, min_years=12, seed=42, outfile_suffix="neg"):
    """
    Create a negatives-only masked dataset by trimming each negative pixel to a
    fake break sampled from the global distribution of positive break times.
    """
    rng  = np.random.RandomState(seed)
    time = ews_ds["time"].values

    def _idx_to_time(cp_idx_da: xr.DataArray) -> xr.DataArray:
        cp = cp_idx_da.values.astype(int)
        out = np.full(cp.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = (cp >= 0) & (cp < time.size)
        out[valid] = time[cp[valid]]
        return xr.DataArray(out, dims=cp_idx_da.dims, coords=cp_idx_da.coords)

    pettitt_sig = (ds_breaks["pettitt_pval"] < alpha) & (ds_breaks["pettitt_cp"] > 0)
    stc_sig     = (ds_breaks["Fstat_pval"]   < alpha) & (ds_breaks["strucchange_bp"] > 0)
    var_sig     = (ds_breaks["pval_var"]     < alpha) & (ds_breaks["bp_var"] > 0)

    bt_pettitt = _idx_to_time(ds_breaks["pettitt_cp"])
    bt_stc     = _idx_to_time(ds_breaks["strucchange_bp"])
    bt_var     = _idx_to_time(ds_breaks["bp_var"])

    t0 = pd.to_datetime(time[0])
    t_thresh = np.datetime64(t0 + pd.DateOffset(years=min_years))

    pettitt_pos = pettitt_sig & (bt_pettitt >= t_thresh)
    stc_pos     = stc_sig     & (bt_stc     >= t_thresh)
    var_pos     = var_sig     & (bt_var     >= t_thresh)

    bt_all = xr.concat(
        [bt_pettitt.where(pettitt_pos),
         bt_stc.where(stc_pos),
         bt_var.where(var_pos)],
        dim="band",
    ).min(dim="band")
    all_pos = (pettitt_pos | stc_pos | var_pos) & bt_all.notnull()

    have_data = ews_ds[var_name].notnull().any(dim="time")

    tests = [
        ("pettitt", pettitt_pos, bt_pettitt),
        ("stc",     stc_pos,     bt_stc),
        ("var",     var_pos,     bt_var),
        ("all",     all_pos,     bt_all),
    ]

    output_dir = Path(output_dir)

    for label, pos_mask, bt in tests:
        neg_mask = have_data & (~pos_mask)

        pool_times = bt.where(pos_mask).values.ravel()
        pool_times = pool_times[~np.isnat(pool_times)]
        pool_times = pool_times[pool_times >= t_thresh]

        if pool_times.size == 0:
            earliest_idx = int(np.searchsorted(time, t_thresh, side="left"))
            fallback_idx = max(earliest_idx, min(int(len(time) * 0.7), len(time) - 1))
            pool_times = np.array([time[fallback_idx]], dtype=time.dtype)

        pseudo_time = xr.full_like(neg_mask, fill_value=np.datetime64("NaT"), dtype="datetime64[ns]")
        ii, jj = np.where(neg_mask.values)
        if ii.size > 0:
            choices = rng.choice(pool_times, size=ii.size, replace=True)
            pseudo_time.values[ii, jj] = choices

        tcoord = ews_ds["time"]
        time_mask_neg = tcoord <= pseudo_time
        sm_neg = ews_ds[[var_name]].where(neg_mask)
        ds_masked_neg = sm_neg.where(time_mask_neg)

        out_path = output_dir / f"{var_name}_cp_masked_{label}_{outfile_suffix}.zarr"
        ds_masked_neg.to_zarr(str(out_path), mode="w")


def main():
    p = argparse.ArgumentParser(description="Mask EWS dataset using pixels with changepoints.")
    p.add_argument("--ews_ds_path", type=str, required=True, help="Path to the EWS output (zarr dir).")
    p.add_argument("--var", type=str, required=True, help="Variable name (Et, precip, sm).")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs.")
    p.add_argument("--cp_ds_path", type=str, default=None,
                   help="Optional override path to changepoint dataset (defaults to inferred *_chp.zarr).")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")

    args = p.parse_args()
    cfg = load_config(args.config)

    ews_ds_path = args.ews_ds_path
    var_name = args.var
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cp_ds_path = args.cp_ds_path or infer_cp_path(ews_ds_path)

    ews_ds = xr.open_dataset(ews_ds_path)
    ds_breaks = xr.open_dataset(cp_ds_path)

    # --- Land mask for precipitation ---
    if var_name == "precip":
        mask_path = cfg_path(cfg, "resources.landsea_mask_grib", must_exist=True)
        ds_mask = xr.open_dataset(mask_path, engine="cfgrib")

        ds_mask = ds_mask.rio.write_crs("EPSG:4326")
        ds_mask = ds_mask.assign_coords(
            longitude=(((ds_mask.longitude + 180) % 360) - 180)
        ).sortby("longitude")
        ds_mask.rio.set_spatial_dims("longitude", "latitude", inplace=True)

        ds_mask = ds_mask.interp(
            longitude=ews_ds.lon,
            latitude=ews_ds.lat,
            method="nearest",
        )

        mask = ds_mask["lsm"] > 0.7
        ews_ds = ews_ds.where(mask)

    mask_breakpoints(ews_ds, var_name, ds_breaks, str(output_dir))
    mask_negatives(ews_ds, var_name, ds_breaks, str(output_dir))


if __name__ == "__main__":
    main()