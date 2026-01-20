To run the demo: 

```
git clone https://github.com/romlotch/25_09_water_resilience_ews
cd 25_09_water_resilience_ews
```
(macOS/linux)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ls demo

# optionally delete old demo outputs if they exist
# rm -rf demo/data demo/outputs
# mkdir -p demo/data demo/outputs
```


(Windows)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ls demo
# optionally delete old demo outputs if they exist
Remove-Item -Recurse -Force demo\data, demo\outputs -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force demo\data | Out-Null
New-Item -ItemType Directory -Force demo\outputs | Out-Null
```

Generate demo dataset and synthetic resources 

```
python demo/make_demo_data.py
python demo/make_demo_resources.py

python -c "from pathlib import Path; print('demo/data:', [p.name for p in Path('demo/data').iterdir()]); print('demo/resources:', [p.name for p in Path('demo/resources').iterdir()])"
```

Run the core pipeline
```
python 01-run_ews.py \
  --dataset demo/data/demo_sm.zarr \
  --variable sm \
  --out demo_sm \
  --freq W \
  --config demo/config.demo.yaml

# find the file output folder that was created:
python -c "from pathlib import Path; p=Path('outputs'); print([x.name for x in p.iterdir()])"

python 01a-combine_ews_output.py \
  --output_dir outputs/processed_tiles/demo_sm \ 
  --variable sm \
  --suffix demo \
  --config demo/config.demo.yaml

# list merged outputs
python -c "from pathlib import Path; print([str(x) for x in Path('outputs/zarr').glob('*.zarr')])"

python 02-run_kt.py \
  --input outputs/zarr/out_sm_demo.zarr \
  --workers 1 \
  --config demo/config.demo.yaml

# list outputs
python -c "from pathlib import Path; print([str(x) for x in Path('outputs/zarr').glob('*.zarr')])"

python 02a-plot_kt.py --dataset outputs/zarr/out_sm_demo_kt.zarr --var sm --config demo/config.demo.yaml
python 02b-plot_biomes.py --dataset outputs/zarr/out_sm_demo_kt.zarr --var sm --config demo/config.demo.yaml

# find the figures
python -c "from pathlib import Path; p=Path('outputs'); figs=list(p.rglob('*.png'))+list(p.rglob('*.svg'))+list(p.rglob('*.pdf')); print([str(x) for x in figs])"





