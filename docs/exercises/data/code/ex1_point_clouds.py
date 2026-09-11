"""Exercise 1 — Point clouds: geometry and spread in 2D."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from report_utils import save_table

# Parameters given by the statement (one row per class).
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def clouds_from_noise(z, s):
    """Build the 4 clouds at scale factor s from standard-normal noise z.

    z has shape (4, N_PER_CLASS, 2). Each point is mu_k + s * sigma_k * z,
    which is exactly a sample of N(mu_k, (s * sigma_k)^2) per axis.
    Reusing the same z for every s means the datasets differ ONLY in their
    spread (the means never change), so the comparison across s is honest.
    """
    X = MEANS[:, None, :] + s * STDS[:, None, :] * z
    y = np.repeat(np.arange(4), z.shape[1])
    return X.reshape(-1, 2), y


def nearest_center(X):
    """Index of the closest class mean (Euclidean) for every point."""
    d = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    return d.argmin(axis=1)


def mixing_rate(X, y):
    """Fraction of points whose nearest class center is not their own."""
    return float(np.mean(nearest_center(X) != y))


def bayes_labels(P, s):
    """Most likely class of each point under the TRUE Gaussian parameters.

    Nothing is trained: this uses the known means/stds. It is the best any
    classifier (a neural network included) could do, so its boundary is
    what a well-trained network would approximate.
    """
    sig = s * STDS
    z2 = (((P[:, None, :] - MEANS[None]) / sig[None]) ** 2).sum(axis=2)
    log_lik = -0.5 * z2 - np.log(sig).sum(axis=1)[None]
    return log_lik.argmax(axis=1)


def separation_ratios():
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) at s = 1."""
    sigma_bar = STDS.mean(axis=1)
    rows = []
    for i, j in itertools.combinations(range(4), 2):
        dist = np.linalg.norm(MEANS[i] - MEANS[j])
        r = dist / (sigma_bar[i] + sigma_bar[j])
        rows.append({"pair": f"({i}, {j})", "dist_means": dist,
                     "sigma_bar_i": sigma_bar[i], "sigma_bar_j": sigma_bar[j],
                     "r_ij (s=1)": r, "r_ij (s=2) = r/2": r / 2})
    return pd.DataFrame(rows)


def scatter_classes(ax, X, y, s=None, size=12):
    for k in range(4):
        ax.scatter(*X[y == k].T, s=size, alpha=0.6, color=COLORS[k],
                   label=f"Class {k}")
    ax.scatter(*MEANS.T, marker="X", s=160, color="black",
               edgecolor="white", linewidth=1.2, label="Class mean", zorder=5)


def run(rng, out_dir):
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    res = {}

    # ---------------- A: generate the clouds (s = 1) ----------------
    z = rng.standard_normal((4, N_PER_CLASS, 2))
    X, y = clouds_from_noise(z, 1.0)
    res["n_samples"] = int(len(X))
    res["samples_per_class"] = np.bincount(y).tolist()
    res["empirical_means_s1"] = [X[y == k].mean(axis=0).round(3).tolist() for k in range(4)]
    res["empirical_stds_s1"] = [X[y == k].std(axis=0, ddof=1).round(3).tolist() for k in range(4)]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_classes(ax, X, y)
    ax.set(title="Figure 1 — Four Gaussian clouds (s = 1), centers marked",
           xlabel="$x_1$", ylabel="$x_2$")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1.png", dpi=150)
    xlim1, ylim1 = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)

    # ---------------- B: more or less spread out ----------------
    datasets = {s: clouds_from_noise(z, s) for s in SCALES}

    all_pts = np.vstack([d[0] for d in datasets.values()])
    pad = 1.0
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SCALES):
        Xs, ys = datasets[s]
        scatter_classes(ax, Xs, ys, size=8)
        ax.set(title=f"s = {s}  (mixing rate = {mixing_rate(Xs, ys):.2%})",
               xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 2 — Same 4 classes, standard deviations scaled by s "
                 "(shared axes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2.png", dpi=150)
    plt.close(fig)

    # Separation ratios at s = 1.
    ratios = separation_ratios()
    save_table(ratios, tab_dir / "ex1_separation.md", "{:.3f}")
    i_min = ratios["r_ij (s=1)"].idxmin()
    res["separation_ratios_s1"] = dict(zip(ratios["pair"], ratios["r_ij (s=1)"].round(4)))
    res["smallest_r_pair"] = ratios.loc[i_min, "pair"]
    res["smallest_r_s1"] = round(float(ratios.loc[i_min, "r_ij (s=1)"]), 4)
    res["smallest_r_s2"] = round(float(ratios.loc[i_min, "r_ij (s=1)"] / 2), 4)

    # Mixing rate for the 4 required scales (+ the Bayes-rule error, for context).
    min_r = res["smallest_r_s1"]
    rows = []
    for s in SCALES:
        Xs, ys = datasets[s]
        rows.append({"s": s, "mixing rate": mixing_rate(Xs, ys),
                     "mixed points": int(np.sum(nearest_center(Xs) != ys)),
                     "Bayes-rule error": float(np.mean(bayes_labels(Xs, s) != ys)),
                     "smallest r_ij = r_01 / s": min_r / s})
    mix = pd.DataFrame(rows)
    save_table(mix, tab_dir / "ex1_mixing.md", "{:.4f}")
    res["mixing_rate"] = {str(s): round(m, 4) for s, m in zip(mix["s"], mix["mixing rate"])}
    res["bayes_error"] = {str(s): round(m, 4) for s, m in zip(mix["s"], mix["Bayes-rule error"])}

    # Which pairs of classes get mixed at s = 1 (row = true class, col = nearest center).
    conf = pd.crosstab(pd.Series(datasets[1.0][1], name="true class"),
                       pd.Series(nearest_center(datasets[1.0][0]), name="nearest center"))
    res["confusion_s1"] = conf.values.tolist()

    # Dense grid of s (same noise z) to locate where mixing starts.
    s_grid = np.round(np.arange(0.25, 4.0001, 0.05), 2)
    mix_grid = np.array([mixing_rate(*clouds_from_noise(z, s)) for s in s_grid])
    bayes_grid = np.array([np.mean(bayes_labels(clouds_from_noise(z, s)[0], s)
                                   != clouds_from_noise(z, s)[1]) for s in s_grid])
    s_first = float(s_grid[np.argmax(mix_grid > 0)])
    res["first_s_with_mixing_on_grid"] = s_first
    res["smallest_r_at_first_mixing"] = round(min_r / s_first, 4)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_grid, mix_grid, color="tab:purple", lw=1.5, alpha=0.6,
            label="Mixing rate (dense grid of s)")
    ax.plot(s_grid, bayes_grid, color="gray", lw=1.2, ls="--",
            label="Bayes-rule error (true parameters)")
    ax.plot(mix["s"], mix["mixing rate"], "o", ms=9, color="tab:purple",
            label="Mixing rate at s ∈ {0.5, 1, 2, 4}")
    for s, m in zip(mix["s"], mix["mixing rate"]):
        ax.annotate(f"{m:.1%}\n$r_{{min}}$={min_r / s:.2f}", (s, m),
                    textcoords="offset points", xytext=(-10, 12), fontsize=9)
    ax.axvline(s_first, color="tab:red", ls=":", label=f"first s with mixing > 0 (s = {s_first})")
    ax.set(title="Figure 3 — Mixing rate × scale factor s",
           xlabel="scale factor s (multiplies every standard deviation)",
           ylabel="mixing rate (fraction of points)", ylim=(-0.02, max(mix_grid) + 0.1))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3.png", dpi=150)
    plt.close(fig)

    # ---------------- C: sketch of decision boundaries on Figure 1 ----------------
    gx, gy = np.meshgrid(np.linspace(*xlim1, 600), np.linspace(*ylim1, 600))
    grid = np.c_[gx.ravel(), gy.ravel()]
    lab_bayes = bayes_labels(grid, 1.0).reshape(gx.shape)
    lab_linear = nearest_center(grid).reshape(gx.shape)
    res["bayes_error_s1_on_sample"] = round(float(np.mean(bayes_labels(X, 1.0) != y)), 4)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(gx, gy, lab_bayes, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=COLORS, alpha=0.12)
    ax.contour(gx, gy, lab_bayes, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1.6)
    ax.contour(gx, gy, lab_linear, levels=[0.5, 1.5, 2.5], colors="dimgray",
               linewidths=1.2, linestyles="--")
    scatter_classes(ax, X, y)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="black", lw=1.6),
                Line2D([], [], color="dimgray", lw=1.2, ls="--")]
    labels += ["Sketched network boundary (curved)", "Straight-line boundaries (nearest center)"]
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.set(title="Figure 1 (annotated) — Sketched decision boundaries, s = 1",
           xlabel="$x_1$", ylabel="$x_2$", xlim=xlim1, ylim=ylim1)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_boundaries.png", dpi=150)
    plt.close(fig)

    return res
