"""Entry point for the Data exercise report.

Runs the three exercises in order with ONE seeded random generator, so every
number and figure in the report is reproducible.

Usage (from the repository root):
    python docs/exercises/data/code/main.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed

import numpy as np

import ex1_point_clouds
import ex2_nonlinearity
import ex3_spaceship_titanic

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent  # docs/exercises/data/
DATA_PATH = OUT_DIR / "dataset" / "train.csv"


def main():
    # The single RNG of the whole report (technical rule of the activity).
    rng = np.random.default_rng(42)

    (OUT_DIR / "figures").mkdir(exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)

    results = {
        "exercise_1": ex1_point_clouds.run(rng, OUT_DIR),
        "exercise_2": ex2_nonlinearity.run(rng, OUT_DIR),
        "exercise_3": ex3_spaceship_titanic.run(rng, OUT_DIR, DATA_PATH),
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
