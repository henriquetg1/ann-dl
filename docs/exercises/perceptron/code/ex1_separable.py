"""Exercise 1 — Separable data: the case the perceptron was designed for."""
import matplotlib.pyplot as plt
import numpy as np

from common import (COLORS, angle_deg, draw_boundary, finish, mark_misclassified,
                    padded_limits, scatter_classes, two_gaussians, unit)
from perceptron import Perceptron

# Parameters given by the statement.
MEAN0, MEAN1 = [1.5, 1.5], [5.0, 5.0]
COV = [[0.5, 0.0], [0.0, 0.5]]
ETA, ETA_ALT, MAX_EPOCHS = 0.01, 1.0, 100


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def run(rng, out_dir):
    fig_dir = out_dir / "figures"
    res = {}

    # ---------------- A: generate the data ----------------
    X, y = two_gaussians(rng, MEAN0, MEAN1, COV)
    res["n_samples"] = int(len(y))
    res["samples_per_class"] = np.bincount(y).tolist()
    res["empirical_means"] = [r(X[y == k].mean(axis=0), 3) for k in (0, 1)]
    xlim, ylim = padded_limits(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    finish(ax, fig, fig_dir / "fig1.png",
           "Figure 1 — Exercise 1 data (2 × 1000 points, separable)", xlim, ylim)

    # ---------------- B/C: train with eta = 0.01 ----------------
    # The only random draw of the model: w ~ N(0, 0.01^2), b = 0.
    w0 = rng.normal(0, 0.01, size=2)
    res["w0"] = r(w0, 6)

    p = Perceptron(w0, 0.0, eta=ETA)
    hist = p.fit(X, y, MAX_EPOCHS)
    wrong = p.predict(X) != y
    res["eta_0.01"] = {
        "w": r(p.w), "b": round(p.b, 4), "epochs": p.epochs,
        "converged": p.converged, "accuracy": p.accuracy(X, y),
        "misclassified": int(wrong.sum()), "direction": r(unit(p.w)),
        "updates_per_epoch": hist["updates"], "accuracy_per_epoch": hist["accuracy"],
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    draw_boundary(ax, p.w, p.b, xlim, color="black", lw=2,
                  label=r"Boundary $\mathbf{w}\cdot\mathbf{x}+b=0$")
    mark_misclassified(ax, X, wrong)
    finish(ax, fig, fig_dir / "fig2.png",
           f"Figure 2 — Learned boundary, η = {ETA} ({p.epochs} epochs, "
           f"acc = {p.accuracy(X, y):.2%})", xlim, ylim)

    # ---------------- D: re-run with eta = 1.0, same w0 and same data order ----------------
    p_alt = Perceptron(w0, 0.0, eta=ETA_ALT)
    hist_alt = p_alt.fit(X, y, MAX_EPOCHS)
    res["eta_1.0"] = {
        "w": r(p_alt.w), "b": round(p_alt.b, 4), "epochs": p_alt.epochs,
        "converged": p_alt.converged, "accuracy": p_alt.accuracy(X, y),
        "direction": r(unit(p_alt.w)),
        "updates_per_epoch": hist_alt["updates"],
        "accuracy_per_epoch": hist_alt["accuracy"],
    }
    res["angle_between_directions_deg"] = round(angle_deg(p.w, p_alt.w), 3)
    # Where each boundary crosses the diagonal x1 = x2 = t (the line joining the means).
    res["diagonal_crossing"] = {
        "eta_0.01": round(-p.b / p.w.sum(), 4),
        "eta_1.0": round(-p_alt.b / p_alt.w.sum(), 4),
    }
    # How big w0 is relative to one update of each run (|x| ~ 5 on average here).
    mean_norm_x = float(np.linalg.norm(X, axis=1).mean())
    res["mean_norm_x"] = round(mean_norm_x, 3)
    res["norm_w0"] = round(float(np.linalg.norm(w0)), 5)

    # Figure 3: accuracy per epoch (both runs), starting from the initial
    # weights (epoch 0); updates per epoch on a second axis.
    acc0 = Perceptron(w0).accuracy(X, y)
    res["initial_accuracy"] = acc0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0] + hist["update_epoch"], [acc0] + hist["update_accuracy"],
            color="tab:purple", lw=0.8, alpha=0.5,
            label=f"Accuracy after each update, η = {ETA}")
    ax.plot([0] + hist["epoch"], [acc0] + hist["accuracy"], "o-", color="tab:purple",
            label=f"Accuracy at end of epoch, η = {ETA}")
    ax.plot([0] + hist_alt["epoch"], [acc0] + hist_alt["accuracy"], "s--",
            color="tab:green", label=f"Accuracy at end of epoch, η = {ETA_ALT}")
    ax2 = ax.twinx()
    ax2.bar(np.array(hist["epoch"]) - 0.15, hist["updates"], width=0.3, alpha=0.25,
            color="tab:purple", label=f"Updates per epoch, η = {ETA}")
    ax2.bar(np.array(hist_alt["epoch"]) + 0.15, hist_alt["updates"], width=0.3,
            alpha=0.25, color="tab:green", label=f"Updates per epoch, η = {ETA_ALT}")
    ax2.set_ylabel("updates (mistakes) in the epoch")
    n_ep = max(p.epochs, p_alt.epochs)
    ax.set(title="Figure 3 — Accuracy × epoch (Exercise 1)", xlabel="epoch",
           ylabel="accuracy on the full dataset", xticks=range(0, n_ep + 1),
           ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3.png", dpi=150)
    plt.close(fig)

    # Supplementary figure: the two boundaries (eta = 0.01 vs 1.0) together.
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    draw_boundary(ax, p.w, p.b, xlim, color="black", lw=2, label=f"η = {ETA}")
    draw_boundary(ax, p_alt.w, p_alt.b, xlim, color="tab:red", lw=2, ls="--",
                  label=f"η = {ETA_ALT}")
    finish(ax, fig, fig_dir / "figS1_eta.png",
           "Figure S1 — Same data, same w₀: boundaries for η = 0.01 and η = 1.0",
           xlim, ylim)

    # Numerical check of the zero-start argument (no random draw involved).
    z1 = Perceptron(np.zeros(2), 0.0, eta=ETA)
    z1.fit(X, y, MAX_EPOCHS)
    z2 = Perceptron(np.zeros(2), 0.0, eta=ETA_ALT)
    z2.fit(X, y, MAX_EPOCHS)
    res["zero_start"] = {
        "eta_0.01": {"w": r(z1.w, 6), "b": round(z1.b, 6), "epochs": z1.epochs},
        "eta_1.0": {"w": r(z2.w, 6), "b": round(z2.b, 6), "epochs": z2.epochs},
        "ratio_w": r(z2.w / z1.w, 6), "ratio_b": round(z2.b / z1.b, 6),
    }
    return res
