"""Data generation and plotting helpers shared by both exercises."""
import matplotlib.pyplot as plt
import numpy as np

N_PER_CLASS = 1000
COLORS = ["tab:blue", "tab:orange"]


def two_gaussians(rng, mean0, mean1, cov):
    """1000 samples per class from N(mean_k, cov), shuffled once.

    Classes are drawn in order (class 0, then class 1) and then shuffled with
    the same rng, so the online training loop does not see 1000 samples of
    one class followed by 1000 of the other. The order is then fixed for every
    epoch and every run, which keeps the runs comparable.
    """
    X0 = rng.multivariate_normal(mean0, cov, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(mean1, cov, size=N_PER_CLASS)
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], N_PER_CLASS)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def scatter_classes(ax, X, y, size=10, alpha=0.5):
    for k in (0, 1):
        ax.scatter(*X[y == k].T, s=size, alpha=alpha, color=COLORS[k],
                   label=f"Class {k}")


def mark_misclassified(ax, X, wrong, label="Misclassified"):
    ax.scatter(*X[wrong].T, s=40, facecolors="none", edgecolors="black",
               linewidths=1.0, label=f"{label} ({int(wrong.sum())})", zorder=4)


def draw_boundary(ax, w, b, xlim, **kw):
    """Line w1*x1 + w2*x2 + b = 0, drawn across the current x range."""
    xs = np.linspace(*xlim, 200)
    if abs(w[1]) > 1e-12:
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    else:  # vertical line
        ax.axvline(-b / w[0], **kw)


def finish(ax, fig, path, title, xlim, ylim, legend_loc="upper left"):
    ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc=legend_loc, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def padded_limits(X, pad=0.5):
    return ((X[:, 0].min() - pad, X[:, 0].max() + pad),
            (X[:, 1].min() - pad, X[:, 1].max() + pad))


def unit(w):
    return w / np.linalg.norm(w)


def angle_deg(u, v):
    """Angle between two vectors, in degrees."""
    c = np.clip(unit(u) @ unit(v), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
