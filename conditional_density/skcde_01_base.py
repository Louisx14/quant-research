import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# === USER INPUTS ============================================
# ============================================================

x_params = [-0.05, -0.0001, 200]   # [x1, x2, steps] -> SL range
n = 100_000                      # number of draws per SL value

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


def comp_comingvol(foresight=48, df=None, colname="coming_vol"):
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

    # Step 1: bullish from marginal
    bullish_samples = rng.choice(bu, size=n_rows, replace=True)

    # Step 2: assign each draw to a bin
    sample_bins = np.digitize(bullish_samples, bu_edges) - 1
    sample_bins = np.clip(sample_bins, 0, n_bins - 1)

    # Step 3: per-bin batch conditional draw
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

def simulate_sl_sweep(samples, sl_values):
    """
    For each SL value, apply the exit logic to the (precomputed) sample table.

    Exit logic:
      direction == 0:
          if SL > bearish_move (i.e. price fell deeper than SL) -> exit at SL
          else                                                  -> exit at bullish_move
      direction == 1:
          exit at bullish_move

    Returns a DataFrame with one row per SL value containing summary stats.
    """
    direction = samples["direction"].to_numpy()
    bearish   = samples["bearish_move"].to_numpy()
    bullish   = samples["bullish_move"].to_numpy()

    results = []
    for sl in sl_values:
        exit_vals = np.where(
            direction == 0,
            np.where(sl > bearish, sl, bullish),   # short branch
            bullish                                # long branch
        )

        positive_mask = exit_vals > 0
        n_pos = positive_mask.sum()

        results.append({
            "SL":              sl,
            "avg_exit":        exit_vals.mean(),
            "avg_positive_exit": exit_vals[positive_mask].mean() if n_pos else np.nan,
            "p_positive":      n_pos / len(exit_vals),
        })

    return pd.DataFrame(results)


# ============================================================
# === RUN ====================================================
# ============================================================

# 1. Build empirical distribution
df_dist = comp_comingvol(foresight=FORESIGHT, df=df_raw)

# 2. Draw the sample table ONCE
samples = sample_conditional_binned(
    df_dist, n_rows=n, n_bins=N_BINS, p_bullish=P_BULLISH, seed=SEED
)

# 3. Build SL grid and run sweep
sl_grid = np.linspace(x_params[0], x_params[1], x_params[2])
results = simulate_sl_sweep(samples, sl_grid)

print(results.to_string(index=False))

# ============================================================
# === PLOT ===================================================
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(results["SL"], results["avg_exit"], color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("SL")
axes[0].set_ylabel("avg exit value")
axes[0].set_title("Average exit (all trades)")
axes[0].grid(alpha=0.3)

axes[1].plot(results["SL"], results["avg_positive_exit"], color="green")
axes[1].set_xlabel("SL")
axes[1].set_ylabel("avg positive exit")
axes[1].set_title("Average exit (winners only)")
axes[1].grid(alpha=0.3)

axes[2].plot(results["SL"], results["p_positive"], color="crimson")
axes[2].set_xlabel("SL")
axes[2].set_ylabel("P(exit > 0)")
axes[2].set_title("Probability of positive exit")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()