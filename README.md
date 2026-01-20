# Terrestrial water cycle resilience: Code and data pipeline

This repository contains code to reproduce the analyses and figures for: 

R. A. Lotcheris, N. Knecht, L. Wang-Erlansson, J. Rocha. Global assessment of terrestrial water cycle resilience. Preprint DOI: https://eartharxiv.org/repository/view/11409/.

We assess the resilience of green water variables (transpiration, soil moisture, and precipitation) using early warning signals (EWS) and breakpoint detection on remotely sensed time series data. This repository reproduces Figs. 1 - 5, Extended Data Figs A1 and A2, and Supplementary Materials S1 to S14. All data for this publication was retrieved from publicly available, open datasets. Data sources are listed in the manuscript under the heading 'Data Availability'. 

The code is organized as a set of scripts (01–06) that:
1) compute rolling EWS indicators on gridded time series,
2) quantify trends in EWS using Kendall’s tau and additional trend estimators
3) detect abrupt shifts via changepoint / structural break tests,
4) evaluate EWS vs detected shifts (“ground-truth” evaluation),
5) fit an ML model to explain/predict abrupt shifts using environmental drivers.


If you use this code, please cite the paper.


---

## Environment / dependencies

### Python
Tested with Python 3.10.

Key packages used:
- numpy, pandas, xarray, dask, zarr
- scipy, statsmodels
- matplotlib, seaborn
- cartopy (global maps)
- rasterio, rioxarray (geospatial I/O)
- geopandas, shapely (biomes)
- scikit-learn (metrics, utilities)
- xgboost + shap (ML script 06b)

### R (required for changepoint analysis)
`03-run_changepoints.py` uses `rpy2` and requires R installed, plus R packages:
- `trend`, `strucchange`, `imputeTS`, `dplyr`, `tibble`, `changepoint`

---

## Setup

```
# Install required packages

python -m venv .venv
source .venv/bin/activate # mac/linux
.venv\Scripts\activate    # windows 
pip install -r requirements.txt

```

## Configuration

Original scripts used hard-coded paths (e.g., large data, land masks, biome shapefiles). To make the repo portable, put all machine-specific paths into a local config file:

1. Copy the example config and edit hardcoded filepaths as needed.

```
cp config.example.yaml config.yaml
```

2. Edit ```config.yaml``` to point to:
- land/sea mask (```resources.landsea_mask_grib```)
- biome shapefile (```resources.tnc_biomes_shapefile```)
- monthly precip + PET inputs (for aridity; ```resources.era5_precip_monthly_nc```, ```resources.pet_monthly_zarr```)
- locations of produced Zarr outputs (```datasets.*```)

Note: config.yaml is not tracked by git. 


## Reproduction workflow

The workflow is run per variable: ```sm```, ```Et```, ```precip```. The workflow here is described using sm as an example. 

### 1. Compute rolling EWS per tile

```
   python3 01-run_ews.py --dataset <PATH> --variable <NAME> --config config.yaml

   # Outputs are saved to: <RUN_NAME>/*.zarr
```
### 2. Merge tiles: 

```
   # Outputs from step 1 inferred 
   python3 01a-combine_ews_output.py --run <RUN_NAME> --variable <VAR> --config config.yaml

   # Output path is defined inside 01a-* (edit or use config)
```

### 3. Plot EWS delta maps (S3):
   
```
python3 01b-plot_deltas.py --variable <VAR> --config config.yaml
```

### 4. Sensitivity analysis (S1):

```
python3 01c-sensitivity.py --variable <VAR> --config config.yaml
# Outputs: 
#    Kappa (agreement) matrices per indicator
#    Light's kappa summaries
#    Kendall tau summaries per configuration 

```

### 5. Compute Kendall tau:

```
python 02-run_kt.py --variable <VAR> --config config.yaml
# Output:
   out_sm_kt.zarr
```

### 6. Plot Kendall Tau and biome-level summaries (Figs. 2 - 4, S2):

```
python 02a-plot_kt.py --var <VAR> --config config.yaml
python 02b-plot_biomes.py --var <VAR> --config config.yaml

```

### 7. Abrupt shift (changepoint) detection and plotting (Fig. 1):

Repeat the pre-break / pre–pseudo-break reruns for whichever breakpoint test you analyze (stc, pettitt, or var).

```
python 03-run_changepoints.py --var <VAR> --config config.yaml
python 03a-plot_changepoints.py --var <VAR> --config config.yaml

```

Plot example time series with abrupt shifts (Fig. 6): 

```
python3 03b-plot_example_abrupt_shift.py --var <VAR> --config config.yaml
```

Plot cummulative area with abrupt shifts (S7; all variables): 

```
python 03c-plot_cumulative_abrupt_shift.py --var <VAR> --config config.yaml
```

### 8. Alternative trend metrics (Theil-Sen and mean change) + agreement (S4, S5, S6): 

```
python 04-run_theil_sen.py --var <VAR> --config config.yaml
python 04a-plot_theil_sen.py --var <VAR> --config config.yaml
```

```
python 04b-run_mean_change.py --var <VAR> --config config.yaml
python 04c-plot_mean_change.py --var <VAR> --config config.yaml
```

```
04d-agreement.py --config config.yaml
```
   
### 9. Breakpoint masking + true positive evaluation (ML pre-requisites): 

Create masked raw datasets (positives = true breakpoints; negatives = pseudo-breakpoints):

```
python 05-mask_breakpoints.py --var <VAR> --config config.yaml
```

This writes (in --out_dir), for each breakpoint test and combined: 

Positives (real breakpoints; late-enough + significant)

- <var>_cp_masked_pettitt.zarr
- <var>_cp_masked_stc.zarr
- <var>_cp_masked_var.zarr
- <var>_cp_masked_all.zarr
- 
Negatives trimmed at pesudo-breakpoints (pseudo-break times sampled from the positive breakpoint-time distribution; seed fixed in script; suffix default is "neg"):

- <var>_cp_masked_pettitt_neg.zarr
- <var>_cp_masked_stc_neg.zarr
- <var>_cp_masked_var_neg.zarr

**Important**: these outputs contain the raw variable time series only (trimmed to ≤ breakpoint time). You must rerun EWS on them. 
Repeat steps 1-3 for both the positives and negatives. Use the CLIs detailed in the scripts to overwrite default file paths: 

#### Positives:

```
# EWS 
python 01-run_ews.py \
  --dataset <OUTPUT_DIR>/<var>_cp_masked_stc.zarr \
  --variable <var> \
  --freq W \
  --out <var>_breakpoints
```
This creates: 
- tiled outputs in `processed_tiles_<var>_breakpoints/`

```
# Merge
python 01a-combine_ews_output.py \
  --output_dir processed_tiles_<var>_breakpoints \
  --variable <var> \
  --suffix breakpoint_stc
```
This produces: 
- `out_<var>_breakpoint_stc.zarr`
  
```
# Kendall Tau
python 02-run_kt.py --input <PATH_TO_out_var_breakpoint_stc.zarr> --workers 8
```
This produces: 
- `out_<var>_breakpoint_stc_kt.zarr` which is --tau_pre

#### Negatives:

```
# EWS 
python 01-run_ews.py \
  --dataset <OUTPUT_DIR>/<var>_cp_masked_stc_neg.zarr \
  --variable <var> \
  --freq W \
  --out <var>_breakpoints_neg
```
This creates: 
- tiled outputs in `processed_tiles_<var>_breakpoints_neg/`

```
# Merge
python 01a-combine_ews_output.py \
  --output_dir processed_tiles_<var>_breakpoints_neg \
  --variable <var> \
  --suffix breakpoint_stc_neg
```
This produces: 
- `out_<var>_breakpoint_stc_neg.zarr`
  
```
# Kendall Tau
python 02-run_kt.py --input <PATH_TO_out_var_breakpoint_stc_neg.zarr> --workers 8
```
This produces: 
- `out_<var>_breakpoint_stc_kt_neg.zarr` which is --tau_pre_neg



### 10a. Build predictor layers: 

```
python 06a-preprocess_rf_drivers.py --config config.yaml
# Output: driver layers as Zarr stores (temperature, precipitation, soil moisture, PET, aridity, ENSO correlation, etc.).
```

### 10b. Fit classifier and plot results (XGBoost + SHAP; Fig. 5, Extended Data A1, A2, S10-14)

The classifier in 06b-run_random_forest.py does not use Kendall’s tau computed on the full time series alone. To avoid leakage and to ensure comparable time support, it uses:
- tau_full: Kendall’s tau computed from EWS indicator time series on the full record (02-run_kt.py run on out_<var>.zarr)
- tau_pre (positives): Kendall’s tau computed from EWS indicator time series recomputed using only the period before the detected breakpoint
- tau_pre_neg (negatives): Kendall’s tau computed from EWS indicator time series recomputed using only the period before a pseudo-breakpoint, where pseudo-breakpoints are sampled from the empirical distribution of positive breakpoint times (per breakpoint test; seed fixed in 05-mask_breakpoints.py)
- 
This means you must run a second EWS + Kendall τ pass for positives and a third EWS + Kendall τ pass for negatives (pseudo-breaks). These are required inputs to the ML script.
```
python3 06b-run_random_forest_scipy.py \
  --var sm \
  --cp_test stc \
  --outdir /mnt/data/romi/figures/paper_1/xgboost_results
```


## Script index (01–06)
EWS + merge:
- 01-run_ews.py (tile EWS computation)
- 01a-combine_ews_output.py (merge tiles)
- 01b-plot_deltas.py (delta maps)
- 01c-sensitivity.py (sensitivity + κ agreement)

Trend metrics:
- 02-run_kt.py + 02a-plot_kt.py (Kendall tau + maps)
- 02b-plot_biomes.py (biome-level bar plot summaries)
- 04-run_theil_sen.py + 04a-plot_theil_sen.py (Theil–Sen slope)
- 04b-run_mean_change.py + 04c-plot_mean_change.py (mean change)
- 04d-agreement.py (agreement; dev)

Abrupt shifts:
- 03-run_changepoints.py (rpy2 + R)
- 03a-plot_changepoints.py (maps)
- 03b-plot_example_abrupt_shift.py (example time series)
- 03c-plot_cumulative_abrupt_shift.py (cumulative area)

Evaluation:
- 05-mask_breakpoints.py (mask positives + pseudo-break negatives)

ML:
- 06a-preprocess_rf_drivers.py (drivers)
- 06b-run_random_forest.py (XGBoost classifier)






