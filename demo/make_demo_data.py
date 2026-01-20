# demo/make_demo_data.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
(DEMO := REPO / "demo").mkdir(exist_ok=True)
(DEMO / "data").mkdir(parents=True, exist_ok=True)
(DEMO / "outputs").mkdir(parents=True, exist_ok=True)
(DEMO / "resources").mkdir(parents=True, exist_ok=True)


def _global_1deg_grid():
    """
    1-degree global grid with cell centers:
      lat: -89.5 .. 89.5 (180)
      lon: -179.5 .. 179.5 (360)
    """
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(-179.5, 180.0, 1.0)
    return lat, lon

def make_demo_dataset(
    var: str = "sm",
    start: str = "2001-01-01",
    end: str = "2010-12-31",
    freq: str = "W",   # weekly (works well with EWS pipelines)
    seed: int = 42,
) -> xr.Dataset:
    rng = np.random.default_rng(seed)

    time = pd.date_range(start=start, end=end, freq=freq)
    lat, lon = _global_1deg_grid()
    T = len(time)
    nlat, nlon = lat.size, lon.size

    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    # -------------------------------------------------------------------------
    # Deterministic background: lat gradient + seasonality (kept mild)
    # -------------------------------------------------------------------------
    t = np.arange(T, dtype="float32")
    tfrac = t / max(1, (T - 1))

    # stronger seasonal amplitude away from equator
    lat_abs = np.abs(lat) / 90.0
    season_amp_lat = (0.03 + 0.10 * lat_abs).astype("float32")  # (lat,)

    seasonal = np.sin(2 * np.pi * t / 52.0).astype("float32")  # weekly annual cycle
    seasonal_3d = (seasonal[:, None, None] * season_amp_lat[None, :, None])

    # baseline mean field: a gentle lat & lon structure
    base = 0.25 + 0.06 * (LAT / 90.0) + 0.02 * (LON / 180.0)  # ~[0.17..0.33]
    base = base.astype("float32")

    # -------------------------------------------------------------------------
    # Define special regions for synthetic "EWS signal" behaviour
    # -------------------------------------------------------------------------
    # A: AC1 and VAR (North America-ish)
    reg_ac1_var_inc = _mask_box(LAT, LON, 40, 55, -120, -90)

    # B: AC1 and VAR (Europe-ish)
    reg_ac1_var_dec = _mask_box(LAT, LON, 45, 60, 0, 25)

    # C: Skew and Kurt (Africa-ish)
    reg_skew_kurt_inc = _mask_box(LAT, LON, -5, 10, 10, 30)

    # D: Mixed (AC1 and VAR) (Asia-ish)
    reg_mixed = _mask_box(LAT, LON, 20, 35, 90, 115)

    # E: flickering
    reg_fd_inc = _mask_box(LAT, LON, -20, -5, -70, -55)

    # No-data 
    reg_nodata = _mask_box(LAT, LON, 2, 6, 32, 36)

    # -------------------------------------------------------------------------
    # Time-varying AR(1) residual simulation
    #
    # Residual process: x_t = phi_t * x_{t-1} + eps_t
    # where phi_t and eps_t vary by region to create EWS patterns.
    # -------------------------------------------------------------------------
    # baseline parameters outside special regions
    phi0 = 0.45
    sig0 = 0.03

    # Pre-allocate output and state arrays
    x_prev = np.zeros((nlat, nlon), dtype="float32")
    field = np.empty((T, nlat, nlon), dtype="float32")

    # for one region maintain a two-state “regime” variable that flips with increasing probability
    state = np.ones((nlat, nlon), dtype="int8")  # +1/-1, only meaningful where reg_fd_inc is True

    for i in range(T):
        f = float(tfrac[i])  # 0..1

        # Start with baseline phi/sigma everywhere
        phi = np.full((nlat, nlon), phi0, dtype="float32")
        sig = np.full((nlat, nlon), sig0, dtype="float32")

        # A: AC1 ↑ and VAR ↑
        phi[reg_ac1_var_inc] = (0.20 + 0.75 * f)          # 0.20 -> 0.95
        sig[reg_ac1_var_inc] = (0.015 + 0.050 * f)        # 0.015 -> 0.065

        # B: AC1 ↓ and VAR ↓
        phi[reg_ac1_var_dec] = (0.90 - 0.70 * f)          # 0.90 -> 0.20
        sig[reg_ac1_var_dec] = (0.060 - 0.045 * f)        # 0.060 -> 0.015

        # D: Mixed (AC1 ↑, VAR ↓)
        phi[reg_mixed] = (0.25 + 0.65 * f)                # 0.25 -> 0.90
        sig[reg_mixed] = (0.055 - 0.035 * f)              # 0.055 -> 0.020

        # Default eps: Gaussian
        eps = (rng.normal(0.0, 1.0, size=(nlat, nlon)).astype("float32") * sig)

        # C: Skew ↑ and Kurt ↑
        #  - increasing skew: exponential component with weight increasing over time
        #  - increasing kurt: heavier-tailed Student-t by decreasing df over time
        if reg_skew_kurt_inc.any():
            df = 30.0 - 27.0 * f  # 30 -> 3 (heavier tails later)
            # Student-t (kurtosis increases as df decreases)
            t_noise = rng.standard_t(df, size=(nlat, nlon)).astype("float32") * sig
            # Skew component (mean-centered exponential)
            skew_w = 0.0 + 0.9 * f
            exp_noise = (rng.exponential(scale=1.0, size=(nlat, nlon)).astype("float32") - 1.0) * sig
            eps_sk = t_noise + skew_w * exp_noise
            eps[reg_skew_kurt_inc] = eps_sk[reg_skew_kurt_inc]

      
        # Increase probability of occasional jumps later in the record.
        if reg_fd_inc.any():
            p_flip = 0.005 + 0.10 * f   # 0.5% -> 10.5% chance per step
            flips = (rng.random((nlat, nlon)) < p_flip) & reg_fd_inc
            state[flips] *= -1

            # regime offset (bimodal)
            regime_amp = (0.00 + 0.12 * f)  # grows over time
            regime_term = regime_amp * state.astype("float32")

            # occasional jumps (more frequent later)
            p_jump = 0.002 + 0.06 * f
            jumps = ((rng.random((nlat, nlon)) < p_jump) & reg_fd_inc).astype("float32")
            jump_sign = rng.choice(np.array([-1.0, 1.0], dtype="float32"), size=(nlat, nlon))
            jump_amp = (0.00 + 0.20 * f)
            jump_term = jumps * jump_sign * jump_amp

            eps[reg_fd_inc] = (eps[reg_fd_inc] + regime_term[reg_fd_inc] + jump_term[reg_fd_inc]).astype("float32")

        # AR(1) update
        x = (phi * x_prev + eps).astype("float32")
        x_prev = x

        # Compose final field: base + season + residual
        field[i, :, :] = (base + seasonal_3d[i, :, :] + x).astype("float32")

   
    # Apply no-data patch (all-NaN for all time) + keep everything else valid
    field[:, reg_nodata] = np.nan

    # keeping values in [0,1]
    if var.lower() in {"sm", "soil_moisture"}:
        field = np.clip(field, 0.0, 1.0)

    ds = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), field)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "title": "Demo dataset for terrestrial water cycle resilience pipeline",
            "note": (
                "Synthetic global 1-degree dataset with region-specific time-evolving "
                "signals designed to trigger EWS patterns (AC1/variance/skew/kurtosis/FD) "
                "and a small all-NaN no-data patch."
            ),
        },
    )

    ds = ds.chunk({"time": min(52, T), "lat": 45, "lon": 90})
    return ds

if __name__ == "__main__":
    ds = make_demo_dataset(var="sm")
    out = DEMO / "data" / "demo_sm.zarr"
    ds.to_zarr(out, mode="w")
    print(f"Wrote {out}")
