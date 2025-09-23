
import re 
import os
import glob
import scipy
import pickle
import argparse
import rasterio 
import rioxarray
import regionmask

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from itertools import product
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import FloatVector

import statsmodels.api as sm

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.interpolate import make_interp_spline

import warnings 
warnings.filterwarnings('ignore')

""" 
Function to run R-based changepoints calculation on .zarr file of variable. 

Assumes R is installed. 

python 03-run_changepoints.py --fp "/mnt/data/romi/output/paper_1/output_Et_final/out_Et.zarr" --var "Et" 
python 03-run_changepoints.py --fp "/mnt/data/romi/output/paper_1/output_sm_final/out_sm.zarr" --var "sm" 
python 03-run_changepoints.py --fp "/mnt/data/romi/output/paper_1/output_persiann_cdr/out_persiann_cdr_weekly_masked.zarr" --var "precip" 

Wraps 'trend' and 'strucchange' R functions. 

Takes around 1 hour 50 min on Gunvor.  

Saves dataset to source path with _chp suffix. 
"""

def prepare_rfunc(): 


    # === Install R packages === 
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  

    for pkg in ["trend", "strucchange", "imputeTS", "dplyr", "tibble", "changepoint"]:

        if not ro.packages.isinstalled(pkg):
            utils.install_packages(pkg)

    ro.r('library(trend)')
    ro.r('library(strucchange)')
    ro.r('library(imputeTS)')
    ro.r('library(dplyr)')
    ro.r('library(tibble)')
    ro.r('library(changepoint)')

    trend = importr("trend")
    strucchange = importr("strucchange")
    imputeTS = importr("imputeTS")
    dplyr = importr("dplyr")
    tibble = importr("tibble")
    changepoint = importr("changepoint")

    # -- Define R function ---
    

    r_code = """
    function(x) {
    library(trend)
    library(strucchange)
    library(imputeTS)
    library(dplyr)
    library(tibble)
    library(changepoint)

    x[x > 900] <- NA
    if(sum(is.na(x)) > 50 || var(x, na.rm=TRUE) == 0) {
        return(rep(NA, 19))
    }
    

    df <- data.frame(clo = x, id = seq_along(x))
    df$clo_lag1 <- c(NA, head(df$clo, -1)) 
    df <- df[complete.cases(df), ]

    dfm <- tibble(clo = x) %>%
        mutate(clo = case_when(clo > 900 ~ NA_real_, TRUE ~ clo)) %>%
        mutate(clo = na_interpolation(clo, option = "spline")) %>%
        mutate(id = row_number()) %>%
        mutate(clo_lag1 = lag(clo))
    
    # Pettitt test
    ptt <- pettitt.test(dfm$clo)
    ptt_cp <- as.integer(ptt$estimate[1])
    ptt_pval <- ptt$p.value

    m1_pettitt <- dfm %>% filter(id < ptt_cp) %>% summarize(mean = mean(clo)) %>% pull(mean)
    m2_pettitt <- dfm %>% filter(id > ptt_cp) %>% summarize(mean = mean(clo)) %>% pull(mean)

    prop_pettitt <- if (!is.na(m1_pettitt) && m1_pettitt != 0) m2_pettitt / m1_pettitt else NA
    diff_pettitt <- if (!is.na(m1_pettitt) && !is.na(m2_pettitt)) m2_pettitt - m1_pettitt else NA

    
    # Structural change
    qlr <- Fstats(clo ~ clo_lag1, data = dfm)
    bps <- breakpoints(qlr)
    sct <- sctest(qlr, type = "supF")
    bp1 <- ifelse(length(bps$breakpoints) > 0, bps$breakpoints[1], NA)
    fstat <- unname(sct$statistic)
    f_pval <- sct$p.value

    m1_stc <- dfm %>% filter(id < bp1) %>% summarize(mean = mean(clo)) %>% pull(mean)
    m2_stc <- dfm %>% filter(id > bp1) %>% summarize(mean = mean(clo)) %>% pull(mean)

    prop_stc <- if (!is.na(m1_stc) && m1_stc != 0) m2_stc / m1_stc else NA
    diff_stc <- if (!is.na(m1_stc) && !is.na(m2_stc)) m2_stc - m1_stc else NA

    # Change in variance
    var_cp_result <- cpt.var(dfm$clo, method = "PELT", penalty = "MBIC", Q = 1)
    var_cpts <- cpts(var_cp_result)
    bp_var <- if (length(var_cpts) > 0 && var_cpts[1] < length(dfm$clo)) var_cpts[1] else NA

    var1_var <- if (!is.na(bp_var)) dfm %>% filter(id < bp_var) %>% summarize(var = var(clo)) %>% pull(var) else NA
    var2_var <- if (!is.na(bp_var)) dfm %>% filter(id > bp_var) %>% summarize(var = var(clo)) %>% pull(var) else NA

    prop_var <- if (!is.na(var1_var) && var1_var != 0) var2_var / var1_var else NA
    diff_var <- if (!is.na(var1_var) && !is.na(var2_var)) var2_var - var1_var else NA

    if (!is.na(bp_var) && bp_var > 2 && bp_var < length(dfm$clo) - 1) {
        seg1 <- dfm$clo[1:(bp_var - 1)]
        seg2 <- dfm$clo[(bp_var + 1):length(dfm$clo)]

        if (length(seg1) >= 2 && length(seg2) >= 2) {
            var_test <- var.test(seg1, seg2)
            pval_var <- var_test$p.value
        } else {
            pval_var <- NA
        }
    } else {
        pval_var <- NA
    }

    return(c(
        pettitt_cp = ptt_cp,
        pettitt_pval = ptt_pval,
        strucchange_bp = bp1,
        Fstat = fstat,
        Fstat_pval = f_pval,
        m1_pettit = m1_pettitt,
        m2_pettitt = m2_pettitt,
        prop_pettitt = prop_pettitt,
        diff_pettitt = diff_pettitt,
        m1_stc = m1_stc,
        m2_stc = m2_stc,
        prop_stc = prop_stc,
        diff_stc = diff_stc,
        bp_var = bp_var,
        pval_var = pval_var,
        var1_var = var1_var,
        var2_var = var2_var,
        prop_var = prop_var,
        diff_var = diff_var
    ))

    }
    """
    r_func = ro.r(r_code)

    return r_func


def run_chp_joblib(fp, var, r_func): 

    # --- Wrapper for R function ---
    def run_structural_tests(ts):
        ts = pd.Series(ts)
        try:
            ts_r = FloatVector(ts)
            res = r_func(ts_r)
            arr = np.array(res)
            if arr.shape != (19,):
                return np.full(19, np.nan)
            return arr
        except Exception as e:
            return np.full(19, np.nan)
        

    print('--- Loading dataset ---')
    ds = xr.open_dataset(f'{fp}')
    Et = ds[f'{var}']

    print('--- Preparing for parallel execution ---')
    lat_vals = Et['lat'].values
    lon_vals = Et['lon'].values
    nlat, nlon = len(lat_vals), len(lon_vals)

    Et_stacked = Et.stack(pixel=('lat', 'lon')).transpose('pixel', 'time')
    time_series_list = [Et_stacked.isel(pixel=i).values for i in range(Et_stacked.sizes['pixel'])]

    print('--- Running ---')
    with tqdm_joblib(tqdm(desc="Processing pixels...", total=len(time_series_list))):
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(run_structural_tests)(ts) for ts in time_series_list
        )

    for i, r in enumerate(results):
        if not isinstance(r, np.ndarray) or r.shape != (19,):
            print(f"Result at index {i} is invalid: {r} of type {type(r)} and shape {getattr(r, 'shape', None)}")

    results_arr = np.array(results)
    print(results_arr.shape)

    results_arr = results_arr.reshape((nlat, nlon, 19))
    var_names = [
            "pettitt_cp",
            "pettitt_pval",
            "strucchange_bp",
            "Fstat",
            "Fstat_pval",
            "m1_pettitt",
            "m2_pettitt",
            "prop_pettitt",
            "diff_pettitt",
            "m1_stc",
            "m2_stc",
            "prop_stc",
            "diff_stc",
            "bp_var",
            "pval_var",
            "m1_var",
            "m2_var",
            "prop_var",
            "diff_var",
        ]

    result_ds = xr.Dataset(
        {var: (("lat", "lon"), results_arr[:, :, i]) for i, var in enumerate(var_names)},
        coords={"lat": lat_vals, "lon": lon_vals}
    )

    input_path = Path(fp)
    output_path = input_path.with_name(input_path.stem + "_chp" + input_path.suffix)

    result_ds.to_zarr(output_path, mode='w')
    print(f"Saved changepoint dataset to: {output_path}")

    return 


def main(fp, var):

    print('--- Preparing R environment ---')
    r_func = prepare_rfunc()

    run_chp_joblib(fp, var, r_func)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run changepoint detection on a dataset.') 
    parser.add_argument('--fp', type=str, required=True, help='Path to the EWS output.') 
    parser.add_argument('--var', type=str, required=True, help='Variable name to process (e.g., Et, precip, sm).')
    
    args = parser.parse_args()

    fp = args.fp
    var = args.var

    main(fp, var)
        

    


