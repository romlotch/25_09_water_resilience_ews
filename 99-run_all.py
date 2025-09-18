import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr



def main(): 

    p = argparse.ArgumentParser()
    p.add_argument('--input_dataset_path',   required = True)
    p.add_argument('--variable_name',  required = True)
    p.add_argument('--output_dir_path', required = True)

    args = p.parse_args()

    input_dataset = args.input_dataset_path ## path to dataset for analysis
    output_dir = args.output_dir_name ## path to output directory 
    var = args.variable_name ## specify the variable on which to run the analysis 


    ## create output directory 

    path = os.path.join(output_dir, f"outputs_{var}")
    os.makedirs(path, exist_ok=True)

    ## run EWS 

    ## run Kendall Tau 
        # Figure 2 

    ## run Thiel Sen 
        # Supp

    ## run breakpoints 
        # Fig 3

    ## mask breakpoints 

    ## create confusion matrix 
        # Fig 4

    ## run EWS on pixels with breakpoints from masked breakpoints 
    ## run KT on EWS of pixels with breakpoints from masked breakpoints

    ## run random forest with pixels with breakpoints as boolean targets, and EWS and environmental variables as predictors 




