"""Exercise 2 — Overlapping data: the case the perceptron cannot solve."""
import math

import matplotlib.pyplot as plt
import numpy as np

from common import (draw_boundary, finish, mark_misclassified, padded_limits,
                    scatter_classes, two_gaussians)
from perceptron import Perceptron

# Parameters given by the statement.
MEAN0, MEAN1 = [3.0, 3.0], [4.0, 4.0]
COV = [[1.5, 0.0], [0.0, 1.5]]
ETA, MAX_EPOCHS = 0.01, 100


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def summarize(X, y, w, b):
    """Accuracy of (w, b) and where its boundary sits relative to the data."""
    y_hat = (X @ w + b >= 0).astype(int)
    center = X.mean(axis=0)
    return {
        "w": r(w), "b": round(float(b), 4),
        "accuracy": float(np.mean(y_hat == y)),
        "share_predicted_class_1": float(y_hat.mean()),
        # Signed distance from the data centroid to the boundary line.
        "centroid_distance": round(float((center @ w + b) / np.linalg.norm(w)), 4),
        # Where the boundary crosses the diagonal x1 = x2 = t (the line joining the means).
        "diagonal_crossing": round(float(-b / w.sum()), 4),
    }


def run(rng, out_dir):
    fig_dir = out_dir / "figures"
    res = {}

    # ---------------- A: generate the data ----------------
    X, y = two_gaussians(rng, MEAN0, MEAN1, COV)
    res["n_samples"] = int(len(y))
    res["samples_per_class"] = np.bincount(y).tolist()
    res["empirical_means"] = [r(X[y == k].mean(axis=0), 3) for k in (0, 1)]
    res["mean_norm_x"] = round(float(np.linalg.norm(X, axis=1).mean()), 3)
    xlim, ylim = padded_limits(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    finish(ax, fig, fig_dir / "fig4.png",
           "Figure 4 — Exercise 2 data (2 × 1000 points, overlapping)", xlim, ylim)

    # Reference: the best straight line for the TRUE distributions is the
    # perpendicular bisector of the means, x1 + x2 = 7 (equal isotropic covariances).
    ref = summarize(X, y, np.array([1.0, 1.0]), -7.0)
    res["reference_line_x1_plus_x2_eq_7"] = ref
    # Its population accuracy: Phi(half the distance between the means / sigma).
    half_gap = np.linalg.norm(np.subtract(MEAN1, MEAN0)) / 2 / math.sqrt(COV[0][0])
    res["best_line_theoretical_accuracy"] = round(0.5 * (1 + math.erf(half_gap / math.sqrt(2))), 4)

    # ---------------- B: train (same class as Exercise 1), with the pocket ----------------
    w0 = rng.normal(0, 0.01, size=2)
    res["w0"] = r(w0, 6)
    p = Perceptron(w0, 0.0, eta=ETA)
    hist = p.fit(X, y, MAX_EPOCHS)
    res["initial_accuracy"] = Perceptron(w0).accuracy(X, y)
    res["epochs_run"] = p.epochs
    res["converged"] = p.converged
    res["final"] = summarize(X, y, p.w, p.b)
    res["pocket"] = summarize(X, y, p.pocket_w, p.pocket_b)
    res["pocket_epoch"] = p.pocket_epoch
    res["updates_per_epoch"] = {"first": hist["updates"][0], "last": hist["updates"][-1],
                                "mean": round(float(np.mean(hist["updates"])), 1)}
    acc = np.array(hist["accuracy"])
    res["current_accuracy_per_epoch"] = {"min": float(acc.min()), "max": float(acc.max()),
                                         "mean": round(float(acc.mean()), 4),
                                         "last_10": acc[-10:].tolist()}
    # Accuracy right after each single update, within the last epoch.
    ue, ua = np.array(hist["update_epoch"]), np.array(hist["update_accuracy"])
    last = ue > MAX_EPOCHS - 1
    res["last_epoch_after_each_update"] = {"min": float(ua[last].min()),
                                           "max": float(ua[last].max()),
                                           "mean": round(float(ua[last].mean()), 4)}
    W = np.array(hist["w"])
    B = np.array(hist["b"])
    res["trajectory"] = {
        "b_first_last": [round(B[0], 4), round(B[-1], 4)],
        "w_norm_min_max": [round(float(np.linalg.norm(W, axis=1).min()), 4),
                           round(float(np.linalg.norm(W, axis=1).max()), 4)],
        "diagonal_crossing_min_max": [round(float((-B / W.sum(axis=1)).min()), 3),
                                      round(float((-B / W.sum(axis=1)).max()), 3)],
    }

    # ---------------- C: Figure 5 (both boundaries) ----------------
    # The final boundary may lie outside the cloud, so widen the view to show it.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    view_x = (min(xlim[0], -1.0), xlim[1])
    view_y = (min(ylim[0], -1.0), ylim[1])
    for ax, (name, w, b) in zip(axes, [("Final", p.w, p.b),
                                       ("Pocket", p.pocket_w, p.pocket_b)]):
        wrong = (X @ w + b >= 0).astype(int) != y
        scatter_classes(ax, X, y, size=8, alpha=0.4)
        mark_misclassified(ax, X, wrong, label=f"Misclassified by {name.lower()}")
        draw_boundary(ax, p.w, p.b, view_x, color="black", lw=2,
                      ls="-" if name == "Final" else ":",
                      label=f"Final boundary (acc = {res['final']['accuracy']:.2%})")
        draw_boundary(ax, p.pocket_w, p.pocket_b, view_x, color="tab:red", lw=2,
                      ls="-" if name == "Pocket" else ":",
                      label=f"Pocket boundary (acc = {res['pocket']['accuracy']:.2%})")
        ax.set(title=f"{name} weights — misclassified points circled",
               xlabel="$x_1$", ylabel="$x_2$", xlim=view_x, ylim=view_y)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 5 — Final vs pocket decision boundaries (Exercise 2)", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5.png", dpi=150)
    plt.close(fig)

    # ---------------- C: Figure 6 (current vs pocket accuracy) ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    # Range of the accuracy right after each update, per epoch (a band: the
    # raw per-update trace has ~77 000 points and is unreadable).
    ep_idx = np.ceil(ue).astype(int)
    lo = [ua[ep_idx == e].min() for e in hist["epoch"]]
    hi = [ua[ep_idx == e].max() for e in hist["epoch"]]
    ax.fill_between(hist["epoch"], lo, hi, color="tab:purple", alpha=0.12, step="mid",
                    label="Min–max accuracy after single updates, within each epoch")
    acc0 = Perceptron(w0).accuracy(X, y)
    ax.plot([0] + hist["epoch"], [acc0] + hist["accuracy"], color="tab:purple", lw=1.2,
            label="Accuracy of the current weights (end of epoch)")
    ax.plot([0] + hist["epoch"], [acc0] + hist["pocket_accuracy"], color="tab:red", lw=2,
            label="Best-so-far (pocket) accuracy")
    ax.axhline(ref["accuracy"], color="gray", ls="--", lw=1,
               label=f"Reference line $x_1+x_2=7$ ({ref['accuracy']:.2%})")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Chance (50%)")
    ax.set(title="Figure 6 — Accuracy × epoch (Exercise 2)", xlabel="epoch",
           ylabel="accuracy on the full dataset", ylim=(0.3, 0.8))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6.png", dpi=150)
    plt.close(fig)

    # Where the boundary cuts the diagonal x1 = x2 = t after EVERY update.
    UW, UB = np.array(hist["update_w"]), np.array(hist["update_b"])
    t_upd = -UB / UW.sum(axis=1)
    second_half = ue > MAX_EPOCHS / 2
    res["crossing_after_each_update_epochs_51_100"] = {
        "median": round(float(np.median(t_upd[second_half])), 3),
        "iqr": r(np.percentile(t_upd[second_half], [25, 75]), 3),
    }
    res["relative_step"] = {  # size of one update compared to the weights themselves
        "eta_norm_x_over_norm_w": round(ETA * res["mean_norm_x"]
                                        / float(np.linalg.norm(UW, axis=1).mean()), 3),
        "eta_over_abs_b": round(ETA / float(np.abs(UB).mean()), 3),
    }

    # Supplementary figure: that crossing over the last 5 epochs.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tail = ue > MAX_EPOCHS - 5
    ax.plot(ue[tail], t_upd[tail], color="black", lw=0.6,
            label="Crossing $t=-b/(w_1+w_2)$ after each update")
    ax.plot(hist["epoch"][-5:], (-B / W.sum(axis=1))[-5:], "o", color="tab:purple",
            label="Snapshot at the end of the epoch")
    ax.axhline(3.5, color="tab:red", ls="--", label="Midpoint of the means (t = 3.5)")
    ax.set(title="Figure S2 — Where the boundary cuts the line $x_1=x_2$ (epochs 96–100)",
           xlabel="epoch (epoch k runs from k − 1 to k)",
           ylabel="t  (boundary passes through (t, t))", ylim=(0, 10))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "figS2_position.png", dpi=150)
    plt.close(fig)

    # ---------------- D: supplementary checks (same data, same w0, no new draws) ----------------
    # They only illustrate the argument made from the update rule in the report.
    checks = {}
    order = np.argsort(y, kind="stable")          # all class 0 first, then class 1
    for name, Xc, yc, eta, epochs in [
        ("sorted by class, eta=0.01, 100 epochs", X[order], y[order], ETA, MAX_EPOCHS),
        ("eta=0.001, 100 epochs", X, y, 0.001, MAX_EPOCHS),
        ("eta=0.01, 500 epochs", X, y, ETA, 500),
    ]:
        pc = Perceptron(w0, 0.0, eta=eta)
        hc = pc.fit(Xc, yc, epochs)
        tail = np.array(hc["accuracy"][-100:])
        checks[name] = {"final_accuracy": pc.accuracy(X, y), "w": r(pc.w), "b": round(pc.b, 4),
                        "pocket_accuracy": pc.pocket_acc, "pocket_epoch": pc.pocket_epoch,
                        "last_100_epochs_mean": round(float(tail.mean()), 4),
                        "last_100_epochs_min_max": [float(tail.min()), float(tail.max())]}
    res["supplementary_checks"] = checks
    return res
