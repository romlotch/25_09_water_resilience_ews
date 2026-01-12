# Terrestrial water cycle resilience: Code and data pipeline

This repository contains code to reproduce the analyses and figures for: 

R. Lotcheris, N. Knecht, L. Wang-Erlansson, J. Rocha. (submitted to Nature Water). Global assessment of terrestrial water cycle resilience. Preprint DOI: https://eartharxiv.org/repository/view/11409/.

This repository reproduces Figs. 1 - 6, Extended Data Figs A1 and A2, and Supplementary Materials S1 to S14. 

All data for this publication was retrieved from publicly available, open datasets. Data sources are listed in the manuscript under the heading 'Data Availability'. 

If you use this code, please cite the paper.

## Reproduction workflow

```
# Install required packages

python -m venv .venv
source .venv/bin/activate # mac/linux
.venv\Scripts\activate    # windows 
pip install -r requirements.txt

```

### Configuration 

The coded uses a local config file for file paths (e.g., large data, land masks, biome shapefiles). 
1. Copy the example config and edit hardcoded filepaths as needed.

```
cp config.example.yaml config.yaml
```

2. Edit ```config.yaml``` to point to:
- land/sea mask (resources.landsea_mask_grib)
- biome shapefile (resources.tnc_biomes_shapefile)
- monthly precip + PET inputs (for aridity; resources.era5_precip_monthly_nc, resources.pet_monthly_zarr)
- locations of produced Zarr outputs (datasets.*)

Note: config.yaml is not tracked by git. 

### Running the pipeline 

1. Compute EWS tiles and merge

```01-run_ews.py``` 
```01a-combine_ews_output.py```

2. Detect abrupt shifts

```03-run_changepoints.py```

3. Compute trend metrics

```02-run_kt.py``` 
```04-run_theil_sen.py```
```04b-run_mean_change.py```
   
4. Plot figures

```01b-plot_deltas.py``` 
```02a-plot_kt.py```
```03c-plot_cumulative_abrupt_shift.py```
etc.

5. Ground-truth evaluation and breakpoint masking

```05-mask_breakpoints.py``` 
```05a-plot_true_positives.py```






