"""Entry point for the Perceptron exercise report.

Runs both exercises in order with ONE seeded random generator, so every
number and figure in the report is reproducible.

Usage (from the repository root):
    python docs/exercises/perceptron/code/main.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed

import numpy as np

import ex1_separable
import ex2_overlapping

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent  # docs/exercises/perceptron/


def main():
    # The single RNG of the whole report (technical rule of the activity).
    rng = np.random.default_rng(42)

    (OUT_DIR / "figures").mkdir(exist_ok=True)

    results = {
        "exercise_1": ex1_separable.run(rng, OUT_DIR),
        "exercise_2": ex2_overlapping.run(rng, OUT_DIR),
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
