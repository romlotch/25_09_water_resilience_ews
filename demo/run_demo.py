# demo/run_demo.py
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    (DEMO / "data").mkdir(exist_ok=True)
    (DEMO / "outputs").mkdir(exist_ok=True)

    # 1) Make demo data
    run(["python", str(DEMO / "make_demo_data.py")])

    run(["python", str(DEMO / "make_demo_resources.py")])

    # 2) Run EWS on the demo dataset
    run([
        "python", str(REPO / "01-run_ews.py"),
        "--dataset", str(DEMO / "data" / "demo_sm.zarr"),
        "--variable", "sm",
        "--config", str(DEMO / "config.demo.yaml"),
    ])

    # 3) Merge tiles (adapt args to your script’s CLI)
    run([
        "python", str(REPO / "01a-combine_ews_output.py"),
        "--run", "demo_sm",              
        "--variable", "sm",
        "--config", str(DEMO / "config.demo.yaml"),
    ])

    # 4) Kendall tau on the merged EWS output
    run([
        "python", str(REPO / "02-run_kt.py"),
        "--variable", "sm",
        "--config", str(DEMO / "config.demo.yaml"),
    ])

    # 5) Make one figure
    run([
        "python", str(REPO / "02a-plot_kt.py"),
        "--var", "sm",
        "--config", str(DEMO / "config.demo.yaml"),
    ])

if __name__ == "__main__":
    main()
