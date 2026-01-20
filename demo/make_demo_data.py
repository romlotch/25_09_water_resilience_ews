# demo/make_demo_data.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

REPO = Path(__file__).resolve().parents[1]
(DEMO := REPO / "demo").mkdir(exist_ok=True)
(DEMO / "data").mkdir(parents=True, exist_ok=True)
(DEMO / "outputs").mkdir(parents=True, exist_ok=True)
(DEMO / "resources").mkdir(parents=True, exist_ok=True)


def _global_1deg_grid(lat_desc: bool = True):
    """
    1-degree global grid at cell centers.

    IMPORTANT: lat_desc=True makes lat run north->south so that
    imshow(origin='upper') plots correctly without flipping.
    """
    lon = np.arange(-179.5, 180.0, 1.0).astype("float64")   # 360
    if lat_desc:
        lat = np.arange(89.5, -90.0, -1.0).astype("float64")  # 180 (descending)
    else:
        lat = np.arange(-89.5, 90.0, 1.0).astype("float64")   # ascending
    return lat, lon



def _mask_box(LAT, LON, lat_min, lat_max, lon_min, lon_max):
    return (LAT >= lat_min) & (LAT <= lat_max) & (LON >= lon_min) & (LON <= lon_max)


def make_demo_dataset(
    var: str = "sm",
    start: str = "2001-01-01",
    end: str = "2015-12-31",
    freq: str = "W",   # weekly 
    seed: int = 42,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)

    time = pd.date_range(start=start, end=end, freq=freq)
    lat, lon = _global_1deg_grid()
    T = len(time)
    nlat, nlon = lat.size, lon.size

    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    # Deterministic background: lat gradient + seasonality
    t = np.arange(T, dtype="float64")
    tfrac = t / max(1, (T - 1))
    seasonal = 0.15 * np.sin(2 * np.pi * t / 52.0) + 0.05 * np.cos(2 * np.pi * t / 52.0)
    drift = 0.0005 * t # small trend
    lat_abs = np.abs(lat) / 90.0
    season_amp_lat = (0.01 + 0.03 * lat_abs).astype("float64")
    seasonal_3d = seasonal[:, None, None] * season_amp_lat[None, :, None]

    base = 0.5 + 0.05 * np.sin(np.deg2rad(LAT)) + 0.03 * np.cos(np.deg2rad(LON))
    noise = rng.normal(0.0, 0.03, size=(T, nlat, nlon))

    # Define special regions for synthetic  EWS
    reg_ac1_var_inc   = _mask_box(LAT, LON, 40, 55, -120, -90)   # A: AC1 VAR up
    reg_ac1_var_dec   = _mask_box(LAT, LON, 45, 60,   0,   25)   # B: AC1 VAR down
    reg_skew_kurt_inc = _mask_box(LAT, LON, -5, 10,  10,   30)   # C: skew kurt up
    reg_mixed         = _mask_box(LAT, LON, 20, 35,  90,  115)   # D: AC1 VAR mixed
    reg_fd_inc        = _mask_box(LAT, LON, -20, -5, -70,  -55)
    
    # No-data 
    reg_nodata = _mask_box(LAT, LON, 2, 6, 32, 36)

    
 
    # Baseline AR(1) residual parameters
    phi0 = 0.35
    sig0 = 0.02

    x_prev = np.zeros((nlat, nlon), dtype="float64")
    field = np.empty((T, nlat, nlon), dtype="float64")

    state = np.ones((nlat, nlon), dtype="int8")

    for i in range(T):
        f = float(tfrac[i])  # 0..1

        phi = np.full((nlat, nlon), phi0, dtype="float64")
        sig = np.full((nlat, nlon), sig0, dtype="float64")

        # A: 
        phi[reg_ac1_var_inc] = (0.25 + 0.65 * f)     # 0.25 -> 0.90
        sig[reg_ac1_var_inc] = (0.015 + 0.055 * f)   # 0.015 -> 0.070

        # B: 
        phi[reg_ac1_var_dec] = (0.90 - 0.60 * f)     # 0.90 -> 0.30
        sig[reg_ac1_var_dec] = (0.070 - 0.050 * f)   # 0.070 -> 0.020

        # D: 
        phi[reg_mixed] = (0.30 + 0.60 * f)           # 0.30 -> 0.90
        sig[reg_mixed] = (0.060 - 0.035 * f)         # 0.060 -> 0.025

        # Default eps: Gaussian
        eps = rng.normal(0.0, 1.0, size=(nlat, nlon)).astype("float64") * sig

        # C:
        if reg_skew_kurt_inc.any():
            df = 20.0 - 17.0 * f      # 20 -> 3 (heavier tails later)
            t_noise = rng.standard_t(df, size=(nlat, nlon)).astype("float64") * sig
            skew_w = 0.0 + 0.9 * f
            exp_noise = (rng.exponential(scale=1.0, size=(nlat, nlon)).astype("float64") - 1.0) * sig
            eps_sk = t_noise + skew_w * exp_noise
            eps[reg_skew_kurt_inc] = eps_sk[reg_skew_kurt_inc]

        # E
        if reg_fd_inc.any():
            p_flip = 0.002 + 0.08 * f
            flips = (rng.random((nlat, nlon)) < p_flip) & reg_fd_inc
            state[flips] *= -1

            regime_amp = 0.00 + 0.10 * f
            regime_term = regime_amp * state.astype("float64")

            p_jump = 0.001 + 0.05 * f
            jumps = ((rng.random((nlat, nlon)) < p_jump) & reg_fd_inc).astype("float64")
            jump_sign = rng.choice(np.array([-1.0, 1.0], dtype="float64"), size=(nlat, nlon))
            jump_amp = 0.00 + 0.18 * f
            jump_term = jumps * jump_sign * jump_amp

            eps[reg_fd_inc] = (eps[reg_fd_inc] + regime_term[reg_fd_inc] + jump_term[reg_fd_inc]).astype("float64")

        # AR(1) update
        x = (phi * x_prev + eps).astype("float64")
        x_prev = x

        # Compose field
        field = base[None, :, :] + seasonal[:, None, None] + drift[:, None, None] + noise
        


    tmp = xr.Dataset(coords={"lat": lat, "lon": lon})
    land = ~regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(tmp).isnull()
    land_vals = land.values
    field[:, ~land_vals] = np.nan
    
    # Apply no-data patch (all-NaN for all time) + keep everything else valid
    field[:, reg_nodata] = np.nan

    # keeping values in [0,1]
    if var.lower() in {"sm", "soil_moisture"}:
        field = 0.05 + 0.90 * (1.0 / (1.0 + np.exp(-field)))
        field = field + rng.normal(0.0, 5e-2, size=field.shape)
        field = np.clip(field, 0.001, 0.999)
        field = field.astype("float64")

    ds = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), field)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "title": "Demo dataset for terrestrial water cycle resilience pipeline",
            "note": "Synthetic global 1° dataset with injected EWS-friendly regional signals + small no-data patch.",
        },
    )

    # Chunking for zarr 
    ds = ds.chunk({"time": min(104, T), "lat": 45, "lon": 90})
    return ds

if __name__ == "__main__":
    ds = make_demo_dataset(var="sm")
    out = DEMO / "data" / "demo_sm.zarr"
    ds.to_zarr(out, mode="w")
    print(f"Wrote {out}")
    print("lat[0], lat[-1]:", float(ds.lat.values[0]), float(ds.lat.values[-1]))
