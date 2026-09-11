"""Exercise 2 — Non-linearity in higher dimensions (5D)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from report_utils import save_table

N = 500  # samples per class
DIM = 5

# Dataset I parameters (from the statement).
MU_A = np.zeros(DIM)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# Dataset II parameters: radius ~ N(mean, std). 0.4 is taken as the standard deviation.
RADIUS_C = (2.0, 0.4)  # core
RADIUS_D = (5.0, 0.4)  # shell
RADIUS_THRESHOLD = 3.5  # midway between the two mean radii


def make_dataset_1(rng):
    XA = rng.multivariate_normal(MU_A, SIGMA_A, size=N)
    XB = rng.multivariate_normal(MU_B, SIGMA_B, size=N)
    return np.vstack([XA, XB]), np.repeat([0, 1], N)


def make_dataset_2(rng):
    # Uniform directions on the unit sphere of R^5: normalize standard-normal vectors.
    v = rng.standard_normal((2 * N, DIM))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    rho = np.concatenate([rng.normal(*RADIUS_C, size=N), rng.normal(*RADIUS_D, size=N)])
    return rho[:, None] * u, np.repeat([0, 1], N), u


def nearest_center_mixing(X, y):
    """Same geometric measure as Exercise 1, with the empirical class means."""
    centers = np.array([X[y == k].mean(axis=0) for k in (0, 1)])
    d = np.linalg.norm(X[:, None, :] - centers[None], axis=2)
    return float(np.mean(d.argmin(axis=1) != y))


def run(rng, out_dir):
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    res = {}

    # ---------------- A and B: generate both datasets ----------------
    X1, y1 = make_dataset_1(rng)
    X2, y2, u = make_dataset_2(rng)

    # Sanity checks: valid covariances, sample covariance close to the target,
    # directions really on the unit sphere.
    res["eigvals_sigma_A"] = np.linalg.eigvalsh(SIGMA_A).round(4).tolist()
    res["eigvals_sigma_B"] = np.linalg.eigvalsh(SIGMA_B).round(4).tolist()
    res["max_abs_cov_error_A"] = round(float(np.abs(np.cov(X1[y1 == 0].T) - SIGMA_A).max()), 4)
    res["max_abs_cov_error_B"] = round(float(np.abs(np.cov(X1[y1 == 1].T) - SIGMA_B).max()), 4)
    res["unit_norm_min_max"] = [round(float(np.linalg.norm(u, axis=1).min()), 6),
                                round(float(np.linalg.norm(u, axis=1).max()), 6)]

    names = {1: "Dataset I — shifted Gaussians", 2: "Dataset II — concentric shells"}
    class_names = {1: ["Class A", "Class B"], 2: ["Class C (core)", "Class D (shell)"]}
    data = {1: (X1, y1), 2: (X2, y2)}

    # ---------------- C: PCA to 2D (Figure 4) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    for ax, k in zip(axes, (1, 2)):
        X, y = data[k]
        pca = PCA(n_components=DIM).fit(X)  # labels are NOT used by PCA
        Z = pca.transform(X)[:, :2]
        evr = pca.explained_variance_ratio_
        for c, color in zip((0, 1), ("tab:blue", "tab:orange")):
            ax.scatter(*Z[y == c].T, s=8, alpha=0.5, color=color, label=class_names[k][c])
        ax.set(title=f"{names[k]}\nPC1+PC2 explain {evr[:2].sum():.1%} of the variance",
               xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        radius = np.linalg.norm(X, axis=1)
        centers = [X[y == c].mean(axis=0) for c in (0, 1)]
        rows.append({
            "dataset": names[k],
            "PC1": evr[0], "PC2": evr[1], "PC1+PC2": evr[:2].sum(),
            "||mu_1 - mu_2|| (5D)": float(np.linalg.norm(centers[0] - centers[1])),
            "mean radius class 1": float(radius[y == 0].mean()),
            "mean radius class 2": float(radius[y == 1].mean()),
            "nearest-center mixing": nearest_center_mixing(X, y),
        })
        res[f"dataset_{k}"] = {
            "explained_variance_ratio": evr.round(4).tolist(),
            "pc1_pc2": round(float(evr[:2].sum()), 4),
            "center_distance_5d": round(rows[-1]["||mu_1 - mu_2|| (5D)"], 4),
            "class_means": [c.round(3).tolist() for c in centers],
            "nearest_center_mixing": round(rows[-1]["nearest-center mixing"], 4),
        }
    fig.suptitle("Figure 4 — PCA projection of each 5D dataset onto its first two components",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    save_table(summary, tab_dir / "ex2_summary.md", "{:.4f}")
    res["dataset_1"]["theoretical_center_distance"] = round(float(np.linalg.norm(MU_B - MU_A)), 4)

    # ---------------- C: radius histograms (Figure 5) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k in zip(axes, (1, 2)):
        X, y = data[k]
        radius = np.linalg.norm(X, axis=1)
        bins = np.linspace(0, radius.max() * 1.02, 50)
        for c, color in zip((0, 1), ("tab:blue", "tab:orange")):
            ax.hist(radius[y == c], bins=bins, alpha=0.55, color=color, label=class_names[k][c])
        if k == 2:
            ax.axvline(RADIUS_THRESHOLD, color="black", ls="--",
                       label=f"$\\|x\\| = {RADIUS_THRESHOLD}$")
        ax.set(title=names[k], xlabel="radius $\\|x\\|$ (computed in 5D)", ylabel="count")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figure 5 — Histogram of the radius $\\|x\\|$ per class", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5.png", dpi=150)
    plt.close(fig)

    # ---------------- D: a simple non-linear function that separates Dataset II ----------------
    # f(x) = ||x||^2 - 3.5^2  ->  f < 0: core (C), f > 0: shell (D). Nothing is trained.
    for k in (1, 2):
        X, y = data[k]
        f = (X ** 2).sum(axis=1) - RADIUS_THRESHOLD ** 2
        acc = float(np.mean((f > 0).astype(int) == y))
        res[f"dataset_{k}"]["radius_rule_accuracy"] = round(acc, 4)
    r2 = np.linalg.norm(X2, axis=1)
    res["dataset_2"]["max_radius_core"] = round(float(r2[y2 == 0].max()), 4)
    res["dataset_2"]["min_radius_shell"] = round(float(r2[y2 == 1].min()), 4)

    return res
