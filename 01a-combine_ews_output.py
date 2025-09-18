import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr

""" 
Combine processed EWS tiles into one dataset for further analysis 

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


def merge_ews_tiles(output_dir: str, var: str, suffix: str | None = None) -> None:
    # collect tile stores
    stores = sorted(glob.glob(os.path.join(output_dir, "*.zarr")))
    if not stores:
        raise FileNotFoundError(f"No .zarr stores found in: {output_dir}")

    tiles = []
    for store in stores:
        ds_tile = xr.open_zarr(store)
        # light rechunk to avoid too many small chunks per tile
        ds_tile = ds_tile.chunk({"lat": 50, "lon": 50, "time": -1})
        tiles.append(ds_tile)

    # merge along coords
    ds = xr.combine_by_coords(tiles, combine_attrs="drop_conflicts")
    ds = ds.chunk({"lat": -1, "lon": -1, "time": -1})

    sfx = _format_suffix(suffix)
    out_path = f"/mnt/data/romi/output/paper_1/output_{var}_final/out_{var}{sfx}.zarr"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Writing: {out_path}")
    ds.to_zarr(out_path, mode="w")
    print("Done!")


def main():
    p = argparse.ArgumentParser(description="Merge per-tile Zarr EWS outputs into one dataset.")
    p.add_argument("--output_dir", required=True, help="Directory containing tile .zarr stores.")
    p.add_argument("--variable",   required=True, help="Variable key (e.g., 'sm', 'Et', 'precip').")
    p.add_argument("--suffix",     required=False, default=None,
                   help="Optional filename suffix (e.g., 'kt' -> out_'var'_'breakpoints_stc'.zarr).")
    args = p.parse_args()

    print("Merging...")
    merge_ews_tiles(args.output_dir, args.variable, args.suffix)

if __name__ == "__main__":
    main()