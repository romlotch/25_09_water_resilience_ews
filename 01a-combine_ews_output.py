import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from utils.config import load_config, cfg_path, cfg_get

""" 
Combine processed EWS tiles into one dataset for further analysis. This script is run 
once you have all the tiles processed. 

Out_path is hardcoded

Inputs:
    --output_dir: directory where the processed tiles were saved 
    --var: name of the variable for which EWS were calculated
    --suffix: specify a filename suffix to differentiate full time series runs from postive and negative runs. 

E.g. 
    python 01a-combine_ews_output.py --output_dir '/mnt/data/romi/dev/ews/processed_tiles_Et' --var 'Et'

For the EWS pre-breakpoint positive 
    python 01a-combine_ews_output.py --output_dir 'processed_tiles_Et_breakpoints' --var 'Et' --suffix 'breakpoint_stc'
For the EWS pre-breakpoint negative
    python 01a-combine_ews_output.py --output_dir 'processed_tiles_Et_breakpoints_neg' --var 'Et' --suffix 'breakpoint_stc_neg'

"""

def _format_suffix(sfx: str | None) -> str:
    """Return a filename-ready suffix starting with '_' or empty string if not set."""
    if sfx is None:
        return ""
    sfx = str(sfx).strip()
    if sfx == "":
        return ""
    return sfx if sfx.startswith("_") else f"_{sfx}"


def merge_ews_tiles(output_dir: str, var: str, suffix: str | None = None, cfg=None) -> None:
    if cfg is None:
        raise ValueError("cfg must be provided (load_config(...) result).")

    # collect tile stores
    stores = sorted(glob.glob(os.path.join(output_dir, "*.zarr")))
    if not stores:
        raise FileNotFoundError(f"No .zarr stores found in: {output_dir}")

    tiles = []
    for store in stores:
        ds_tile = xr.open_dataset(store)
        ds_tile = ds_tile.chunk({"lat": 50, "lon": 50, "time": -1})
        tiles.append(ds_tile)

    # merge along coords
    ds = xr.combine_by_coords(tiles, combine_attrs="drop_conflicts")
    for t in tiles:
        t.close()
    ds = ds.chunk({"lat": -1, "lon": -1, "time": -1})

    # output path from config
    sfx = _format_suffix(suffix)

    outputs_root = cfg_path(cfg, "paths.outputs_root", mkdir=True)
    zarr_out_dir = Path(outputs_root) / "zarr"
    zarr_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = zarr_out_dir / f"out_{var}{sfx}.zarr"

    print(f"Writing: {out_path}")
    ds.to_zarr(str(out_path), mode="w")
    print("Done!")


def main():
    p = argparse.ArgumentParser(description="Merge per-tile Zarr EWS outputs into one dataset.")
    p.add_argument("--output_dir", required=False, default=None,
               help="Directory containing processed tile .zarr stores. If omitted, inferred from config + --run.")
    p.add_argument("--run", required=False, default=None,
                help="Run key (matches 01-run_ews.py --out). Used to infer processed tiles folder if --output_dir not given.")
    p.add_argument("--variable", required=True, help="Variable key (e.g., 'sm', 'Et', 'precip').")
    p.add_argument(
        "--suffix",
        required=False,
        default=None,
        help="Optional filename suffix (e.g., 'breakpoint_stc' -> out_<var>_breakpoint_stc.zarr).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = p.parse_args()

    cfg = load_config(args.config)

    print("Merging...")


    outputs_root = cfg_path(cfg, "paths.outputs_root", must_exist=True)
    if args.output_dir is None:
        if args.run is None:
            raise ValueError("Provide either --output_dir or --run (the suffix used in 01-run_ews.py).")
        output_dir = Path(outputs_root) / "processed_tiles" / args.run
    else:
        output_dir = Path(args.output_dir)

    merge_ews_tiles(str(output_dir), args.variable, args.suffix, cfg=cfg)

if __name__ == "__main__":
    main()