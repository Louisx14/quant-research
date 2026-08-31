"""
Conditional distributions of the bearish excursion per bin of the bullish
excursion, with a smoothed kernel density overlay per bin, and the coloured
scatter used to judge how well the bins separate.

Original file: smoothed_kernel_conditional_sampling_empirical.py
"""

import sqlite3

import pandas as pd

import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DB_DIR = Path("data")

DATABASE = "crypto.db"

TABLENAME = "solana1H"

db_path = DB_DIR / DATABASE

conn = sqlite3.connect(db_path)

df_raw = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)

conn.close()


def compute_coming_vol(foresight=48, df=None, colname="coming_vol"):
    df = df.copy()
    colname_bearish = f"{colname}_br"
    colname_bullish = f"{colname}_bu"
    colname_theo_bu = f"{colname}_theo_bu"
    colname_theo_br = f"{colname}_theo_br"

    future_min = df.iloc[:, 5][::-1].rolling(foresight, min_periods=1).min()[::-1].shift(-1)
    df[colname_bearish] = (future_min - df.iloc[:, 5]) / df.iloc[:, 5]

    future_min = df.iloc[:, 4][::-1].rolling(foresight, min_periods=1).min()[::-1].shift(-1)
    df[colname_theo_br] = (future_min - df.iloc[:, 5]) / df.iloc[:, 5]

    future_max = df.iloc[:, 3][::-1].rolling(foresight, min_periods=1).max()[::-1].shift(-1)
    df[colname_theo_bu] = (future_max - df.iloc[:, 5]) / df.iloc[:, 5]

    future_max = df.iloc[:, 5][::-1].rolling(foresight, min_periods=1).max()[::-1].shift(-1)
    df[colname_bullish] = (future_max - df.iloc[:, 5]) / df.iloc[:, 5]

    df = df.dropna()
    return df

def plot_conditional_bearish_grid(df_dist, n_bins=20):

    br = df_dist["coming_vol_br"].to_numpy()
    bu = df_dist["coming_vol_bu"].to_numpy()

    # Quantile-based edges so each bin has roughly equal count
    bu_edges = np.quantile(bu, np.linspace(0, 1, n_bins + 1))
    bu_edges[0]  -= 1e-9
    bu_edges[-1] += 1e-9

    bin_idx = np.digitize(bu, bu_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Shared x-range and bin edges for the histograms so they're visually comparable
    br_min, br_max = br.min(), br.max()
    hist_bins = np.linspace(br_min, br_max, 40)

    # Global y-max: use the densest bin so all subplots share the same vertical scale
    counts_per_bin = [np.histogram(br[bin_idx == k], bins=hist_bins)[0]
                      for k in range(n_bins)]
    y_max = max(c.max() for c in counts_per_bin if len(c)) * 1.1

    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for k in range(n_bins):
        ax = axes[k]
        bearish_in_bin = br[bin_idx == k]
        ax.hist(bearish_in_bin, bins=hist_bins, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, y_max)
        ax.set_title(
            f"Bin {k+1}: bu ∈ [{bu_edges[k]:.3f}, {bu_edges[k+1]:.3f}]\n"
            f"n = {len(bearish_in_bin)}, mean = {bearish_in_bin.mean():.4f}",
            fontsize=9,
        )
        ax.axvline(bearish_in_bin.mean(), color="red", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)

    # Common labels
    fig.supxlabel("bearish move (coming_vol_br)")
    fig.supylabel("count")
    fig.suptitle(
        "Conditional distribution of bearish values, by bullish quantile bin",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    plt.show()


def plot_primary_with_dividers(df_dist, n_bins=20, hist_bins=80):
    bu = df_dist["coming_vol_bu"].to_numpy()

    # Same quantile edges as the conditional grid → n_bins+1 edges, n_bins regions
    bu_edges = np.quantile(bu, np.linspace(0, 1, n_bins + 1))

    fig, ax = plt.subplots(figsize=(14, 6))

    # True shape of the primary distribution (equal-WIDTH bins)
    ax.hist(bu, bins=hist_bins, color="steelblue", edgecolor="white", linewidth=0.3)

    # Red dividers at every quantile cut (equal-COUNT boundaries)
    for edge in bu_edges:
        ax.axvline(edge, color="red", linestyle="--", linewidth=1, alpha=0.8)

    # Number each bin at the top so you can map back to the grid
    ymax = ax.get_ylim()[1]
    for k in range(n_bins):
        centre = (bu_edges[k] + bu_edges[k + 1]) / 2
        ax.text(centre, ymax * 0.97, str(k + 1),
                ha="center", va="top", fontsize=8, color="darkred")

    ax.set_xlabel("bullish move (coming_vol_bu)  —  primary variable")
    ax.set_ylabel("count")
    ax.set_title(
        f"Primary distribution with {n_bins} quantile cuts\n"
        "(each red divider bounds one equal-count bin → its own secondary distribution)",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_conditional_means(df_dist, n_bins=20, show_scatter=True):

    br = df_dist["coming_vol_br"].to_numpy()
    bu = df_dist["coming_vol_bu"].to_numpy()

    bu_edges = np.quantile(bu, np.linspace(0, 1, n_bins + 1))
    bu_edges[0]  -= 1e-9
    bu_edges[-1] += 1e-9

    bin_idx = np.digitize(bu, bu_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bu_bin_means = np.array([bu[bin_idx == k].mean() for k in range(n_bins)])
    br_bin_means = np.array([br[bin_idx == k].mean() for k in range(n_bins)])

    # Vibrant blue -> purple gradient
    cmap = LinearSegmentedColormap.from_list(
        "blue_purple",
        ["#0066ff", "#3b3bff", "#6a1bff", "#a020f0", "#c800c8"],
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    if show_scatter:
        sc = ax.scatter(
            bu, br,
            c=bin_idx, cmap=cmap,
            s=3, alpha=0.45,
            vmin=0, vmax=n_bins - 1,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=ax, ticks=[0, n_bins // 2, n_bins - 1])
        cbar.set_label("bullish bin index")
        cbar.ax.set_yticklabels(["low", "mid", "high"])

    ax.plot(
        bu_bin_means, br_bin_means,
        "o-", color="crimson",
        linewidth=2.2, markersize=3,
        markeredgecolor="white", markeredgewidth=1,
        label="E[bearish | bullish bin]",
        zorder=5,
    )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("mean bullish (per bin)")
    ax.set_ylabel("mean bearish (per bin)")
    ax.set_title(f"Conditional mean of bearish vs bullish, {n_bins} quantile bins")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# --- run ---


if __name__ == "__main__":
    df_dist = compute_coming_vol(foresight=48, df=df_raw)
    plot_conditional_bearish_grid(df_dist, n_bins=20)
    plot_primary_with_dividers(df_dist, n_bins=20)
    plot_conditional_means(df_dist, n_bins=20, show_scatter=True)
