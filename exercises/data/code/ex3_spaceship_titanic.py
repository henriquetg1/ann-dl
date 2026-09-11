"""Exercise 3 — Preparing the Spaceship Titanic data for a tanh network."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from report_utils import save_table

TARGET = "Transported"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM = ["Age"] + SPEND
CAT = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]
LOG_COLS = SPEND + ["TotalSpend"]  # heavy-tailed columns that get log(1 + x)


def load(path):
    df = pd.read_csv(path)
    # Booleans with missing values are read as objects; keep them as strings
    # ("True"/"False") so all categorical columns are handled the same way.
    for c in ["CryoSleep", "VIP"]:
        df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v))
    df[TARGET] = df[TARGET].astype(int)
    return df


def stratified_split(y, test_frac, rng):
    """80/20 split that keeps the class proportions, using the report's RNG."""
    test_idx = []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        test_idx.append(idx[: int(round(test_frac * len(idx)))])
    test_idx = np.sort(np.concatenate(test_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return train_idx, test_idx


class Preprocessor:
    """Every statistic (median, mode, categories, mean, std) is learned in fit()
    on the TRAINING set only; transform() just applies them."""

    def fit(self, df):
        self.num_imputer = SimpleImputer(strategy="median").fit(df[NUM])
        self.cat_imputer = SimpleImputer(strategy="most_frequent").fit(df[CAT])
        self.scaler = StandardScaler().fit(self._numeric(df))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(self._categorical(df))
        return self

    def _numeric(self, df):
        num = pd.DataFrame(self.num_imputer.transform(df[NUM]), columns=NUM, index=df.index)
        num["TotalSpend"] = num[SPEND].sum(axis=1)       # feature engineering
        num[LOG_COLS] = np.log1p(num[LOG_COLS])           # tame the heavy tails
        return num

    def _categorical(self, df):
        return pd.DataFrame(self.cat_imputer.transform(df[CAT]), columns=CAT, index=df.index)

    def transform(self, df):
        num = self._numeric(df)
        num = pd.DataFrame(self.scaler.transform(num), columns=num.columns, index=df.index)
        cat = self.encoder.transform(self._categorical(df))
        cat = pd.DataFrame(cat, columns=self.encoder.get_feature_names_out(CAT), index=df.index)
        return pd.concat([num, cat], axis=1)


def hist_by_class(ax, values, y, bins, title, xlabel):
    for c, color, name in ((0, "tab:blue", "Not transported"), (1, "tab:orange", "Transported")):
        ax.hist(values[y == c], bins=bins, alpha=0.55, color=color, label=name)
    ax.set(title=title, xlabel=xlabel, ylabel="count (passengers)")
    ax.grid(alpha=0.3)
    ax.legend()


def run(rng, out_dir, data_path):
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    res = {}
    df = load(data_path)

    # ---------------- A: get to know the data (descriptive only) ----------------
    res["raw_shape"] = list(df.shape)
    balance = df[TARGET].value_counts().sort_index()
    res["class_counts"] = {"False": int(balance[0]), "True": int(balance[1])}
    res["positive_share"] = round(float(df[TARGET].mean()), 4)

    missing = pd.DataFrame({
        "column": df.columns,
        "missing (count)": df.isna().sum().values,
        "missing (%)": (100 * df.isna().mean()).values,
    })
    save_table(missing, tab_dir / "ex3_missing.md", "{:.2f}")
    res["missing_total_cells"] = int(df.isna().sum().sum())
    res["rows_with_any_missing"] = int(df.isna().any(axis=1).sum())

    spend_stats = pd.DataFrame({
        "column": SPEND,
        "mean": [df[c].mean() for c in SPEND],
        "median": [df[c].median() for c in SPEND],
        "max": [df[c].max() for c in SPEND],
        "share of zeros": [(df[c] == 0).mean() for c in SPEND],
    })
    save_table(spend_stats, tab_dir / "ex3_spend_stats.md", "{:.2f}")

    # ---------------- B: split BEFORE any transformation ----------------
    train_idx, test_idx = stratified_split(df[TARGET].values, 0.20, rng)
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    res["train_rows"], res["test_rows"] = len(train), len(test)
    res["positive_share_train"] = round(float(train[TARGET].mean()), 4)
    res["positive_share_test"] = round(float(test[TARGET].mean()), 4)
    res["foodcourt_train_mean_raw"] = round(float(train["FoodCourt"].mean()), 2)
    res["foodcourt_train_median_raw"] = round(float(train["FoodCourt"].median()), 2)

    # ---------------- C: preprocess (fit on train, apply to both) ----------------
    X_train_raw = train.drop(columns=DROP + [TARGET])
    X_test_raw = test.drop(columns=DROP + [TARGET])
    y_train, y_test = train[TARGET].values, test[TARGET].values

    prep = Preprocessor().fit(X_train_raw)
    X_train = prep.transform(X_train_raw)
    X_test = prep.transform(X_test_raw)

    res["imputation_values"] = {
        **dict(zip(NUM, prep.num_imputer.statistics_.tolist())),
        **dict(zip(CAT, prep.cat_imputer.statistics_.tolist())),
    }
    res["scaler_mean"] = dict(zip(prep.scaler.feature_names_in_, prep.scaler.mean_.round(4).tolist()))
    res["scaler_std"] = dict(zip(prep.scaler.feature_names_in_, prep.scaler.scale_.round(4).tolist()))
    res["onehot_columns"] = prep.encoder.get_feature_names_out(CAT).tolist()

    # A category that never appeared in training ("Pluto") becomes an all-zero block.
    probe = X_test_raw.iloc[[0]].copy()
    probe["HomePlanet"] = "Pluto"
    probe_row = prep.transform(probe)
    res["unseen_category_homeplanet_block"] = probe_row[
        [c for c in probe_row.columns if c.startswith("HomePlanet_")]].iloc[0].tolist()

    # Figure S1: log(1 + x) on one spending column (Spa), training set, before scaling.
    spa_raw = prep.num_imputer.transform(X_train_raw[NUM])[:, NUM.index("Spa")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    hist_by_class(axes[0], spa_raw, y_train, 60, "Before: raw Spa", "Spa (credits)")
    hist_by_class(axes[1], np.log1p(spa_raw), y_train, 60,
                  "After: log(1 + Spa)", "log(1 + Spa)")
    fig.suptitle("Figure S1 — Effect of $\\log(1+x)$ on a heavy-tailed column (Spa, training set)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "figS1_log_spa.png", dpi=150)
    plt.close(fig)

    # ---------------- D: verify and visualize ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fc_raw = X_train_raw["FoodCourt"].values
    hist_by_class(axes[0], fc_raw[~np.isnan(fc_raw)], y_train[~np.isnan(fc_raw)], 60,
                  "Before preprocessing: raw FoodCourt", "FoodCourt (credits)")
    hist_by_class(axes[1], X_train["FoodCourt"].values, y_train, 60,
                  "After preprocessing: impute → log(1 + x) → standardize",
                  "FoodCourt (standardized log scale)")
    fig.suptitle("Figure 6 — FoodCourt before and after preprocessing (training set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6.png", dpi=150)
    plt.close(fig)

    num_cols = NUM + ["TotalSpend"]
    ranges = pd.DataFrame({
        "column": num_cols,
        "train min": X_train[num_cols].min().values,
        "train max": X_train[num_cols].max().values,
        "test min": X_test[num_cols].min().values,
        "test max": X_test[num_cols].max().values,
        "train mean": X_train[num_cols].mean().values,
        "train std": X_train[num_cols].std(ddof=0).values,
    })
    save_table(ranges, tab_dir / "ex3_scaled_ranges.md", "{:.3f}")

    res["final_checks"] = {
        "nan_train": int(X_train.isna().sum().sum()),
        "nan_test": int(X_test.isna().sum().sum()),
        "shape_train": list(X_train.shape),
        "shape_test": list(X_test.shape),
        "train_min": round(float(X_train.values.min()), 4),
        "train_max": round(float(X_train.values.max()), 4),
        "test_min": round(float(X_test.values.min()), 4),
        "test_max": round(float(X_test.values.max()), 4),
        "share_train_values_in_[-3,3]": round(float((np.abs(X_train.values) <= 3).mean()), 4),
        "columns": X_train.columns.tolist(),
    }
    return res
