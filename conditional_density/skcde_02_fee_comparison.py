"""
SKCDE with and without fees — expected value per trade across the stop grid (May 2026).

Original file: SKCDE_002(with fee comparaison).py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# === USER INPUTS ============================================
# ============================================================

x_params = [-0.05, -0.001, 200]   # [x1, x2, steps] -> SL range
n = 100_000                      # number of draws per SL value
FEE = 0.0015                     # 0.15% flat haircut applied once to each exit

DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
FORESIGHT = 48
N_BINS = 20                      # bullish quantile bins for conditional sampling
P_BULLISH = 0.5                  # P(direction = 1)
SEED = 42

# ============================================================
# === DATA LOADING + DISTRIBUTION ============================
# ============================================================

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


# ============================================================
# === CONDITIONAL SAMPLER ====================================
# ============================================================

def sample_conditional_binned(df_dist, n_rows, n_bins=20, p_bullish=0.5, seed=None):
    """
    Draw n_rows samples. Bullish drawn from its historical marginal, bearish drawn
    from the conditional empirical distribution given which bullish bin was selected.
    Direction is an independent coin flip.
    """
    rng = np.random.default_rng(seed)

    br = df_dist["coming_vol_br"].to_numpy()
    bu = df_dist["coming_vol_bu"].to_numpy()

    bu_edges = np.quantile(bu, np.linspace(0, 1, n_bins + 1))
    bu_edges[0]  -= 1e-9
    bu_edges[-1] += 1e-9
    hist_bin_idx = np.digitize(bu, bu_edges) - 1
    hist_bin_idx = np.clip(hist_bin_idx, 0, n_bins - 1)
    bearish_by_bin = [br[hist_bin_idx == k] for k in range(n_bins)]

    bullish_samples = rng.choice(bu, size=n_rows, replace=True)

    sample_bins = np.digitize(bullish_samples, bu_edges) - 1
    sample_bins = np.clip(sample_bins, 0, n_bins - 1)

    bearish_samples = np.empty(n_rows)
    for k in range(n_bins):
        mask = sample_bins == k
        cnt = mask.sum()
        if cnt == 0:
            continue
        pool = bearish_by_bin[k]
        if len(pool) == 0:
            pool = br
        bearish_samples[mask] = rng.choice(pool, size=cnt, replace=True)

    direction = rng.binomial(1, p_bullish, size=n_rows)

    return pd.DataFrame({
        "direction":    direction,
        "bearish_move": bearish_samples,
        "bullish_move": bullish_samples,
    })


# ============================================================
# === SIMULATION =============================================
# ============================================================

def simulate_sl_sweep(samples, sl_values, fee=0.0):
    """
    For each SL value, apply the exit logic to the (precomputed) sample table.

    Exit logic (long-only, direction is a coin flip for which leg comes first):
      direction == 0: if SL > bearish_move -> exit at SL, else exit at bullish_move
      direction == 1: exit at bullish_move

    A flat 'fee' is subtracted once from every exit value before computing stats.

    Returns a DataFrame with one row per SL value.
    """
    direction = samples["direction"].to_numpy()
    bearish   = samples["bearish_move"].to_numpy()
    bullish   = samples["bullish_move"].to_numpy()

    results = []
    for sl in sl_values:
        exit_vals = np.where(
            direction == 0,
            np.where(sl > bearish, sl, bullish),
            bullish
        )
        exit_vals = exit_vals - fee   # apply fee once at the end

        positive_mask = exit_vals > 0
        n_pos = positive_mask.sum()

        results.append({
            "SL":                sl,
            "avg_exit":          exit_vals.mean(),
            "avg_positive_exit": exit_vals[positive_mask].mean() if n_pos else np.nan,
            "p_positive":        n_pos / len(exit_vals),
        })

    return pd.DataFrame(results)


# ============================================================
# === RUN ====================================================
# ============================================================

df_dist = compute_coming_vol(foresight=FORESIGHT, df=df_raw)

samples = sample_conditional_binned(
    df_dist, n_rows=n, n_bins=N_BINS, p_bullish=P_BULLISH, seed=SEED
)

sl_grid = np.linspace(x_params[0], x_params[1], x_params[2])

results_nofee = simulate_sl_sweep(samples, sl_grid, fee=0.0)
results_fee   = simulate_sl_sweep(samples, sl_grid, fee=FEE)

print("=== No fees ===")
print(results_nofee.to_string(index=False))
print(f"\n=== With {FEE*100:.2f}% fee ===")
print(results_fee.to_string(index=False))

# ============================================================
# === PLOT ===================================================
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Average exit ---
axes[0].plot(results_nofee["SL"], results_nofee["avg_exit"], color="steelblue", label="no fees")
axes[0].plot(results_fee["SL"], results_fee["avg_exit"], color="steelblue", alpha=0.6, label=f"with {FEE*100:.2f}% fee")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("SL")
axes[0].set_ylabel("avg exit value")
axes[0].set_title("Average exit (all trades)")
axes[0].grid(alpha=0.3)
axes[0].legend()

# --- Average positive exit ---
axes[1].plot(results_nofee["SL"], results_nofee["avg_positive_exit"], color="green", label="no fees")
axes[1].plot(results_fee["SL"], results_fee["avg_positive_exit"], color="green", alpha=0.6, label=f"with {FEE*100:.2f}% fee")
axes[1].set_xlabel("SL")
axes[1].set_ylabel("avg positive exit")
axes[1].set_title("Average exit (winners only)")
axes[1].grid(alpha=0.3)
axes[1].legend()

# --- Probability of positive exit ---
axes[2].plot(results_nofee["SL"], results_nofee["p_positive"], color="crimson", label="no fees")
axes[2].plot(results_fee["SL"], results_fee["p_positive"], color="crimson", alpha=0.6, label=f"with {FEE*100:.2f}% fee")
axes[2].set_xlabel("SL")
axes[2].set_ylabel("P(exit > 0)")
axes[2].set_title("Probability of positive exit")
axes[2].grid(alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()