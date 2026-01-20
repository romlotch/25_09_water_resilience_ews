# demo/make_demo_data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
(DEMO := REPO / "demo").mkdir(exist_ok=True)
(DEMO / "data").mkdir(parents=True, exist_ok=True)
(DEMO / "outputs").mkdir(parents=True, exist_ok=True)
(DEMO / "resources").mkdir(parents=True, exist_ok=True)

def make_demo_dataset(
    var: str = "sm",
    nlat: int = 12,
    nlon: int = 18,
    start: str = "2001-01-01",
    end: str = "2005-12-01",
    freq: str = "W",  # monthly start
    seed: int = 42,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)

    time = pd.date_range(start=start, end=end, freq=freq)
    lat = np.linspace(-10, 10, nlat)
    lon = np.linspace(-20, 20, nlon)

    T = len(time)

    # Seasonal signal
    month = (np.arange(T) % 12)
    seasonal = 0.2 * np.sin(2 * np.pi * month / 12.0)

    # Base field + noise
    base = rng.normal(loc=0.0, scale=0.08, size=(T, nlat, nlon))
    field = base + seasonal[:, None, None]

    # Add an abrupt shift after a breakpoint time
    bp_idx = int(T * 0.6)  # 60% into record
    shift_mask = np.zeros((nlat, nlon), dtype=bool)
    shift_mask[3:7, 6:12] = True  # a block of pixels shifts
    field[bp_idx:, shift_mask] += 0.35

    ds = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), field.astype("float32"))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "title": "Demo dataset for terrestrial water cycle resilience pipeline",
            "note": "Synthetic data with an abrupt shift in a subset of grid cells.",
        },
    )
    return ds

if __name__ == "__main__":
    ds = make_demo_dataset(var="sm")
    ds.to_zarr("demo/data/demo_sm.zarr", mode="w")
    print("Wrote demo/data/demo_sm.zarr")
