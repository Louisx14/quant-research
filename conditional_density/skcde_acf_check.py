"""
ACF of raw and absolute log returns (May 2026).

Used to choose the block length for the block bootstrap in
`synthetic_markets/`: the absolute-return ACF shows volatility clustering out
to ~125 candles (with a 24h session periodicity), so blocks are set to 250.

Original file: SKCDE_ACFcheck.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# === USER INPUTS ============================================
# ============================================================

DB_DIR    = Path("data")
DATABASE  = "crypto.db"
TABLENAME = "solana1H"

MAX_LAG = 500          # how many lags to show on the ACF plot
CLOSE_COL_IDX = 5      # same convention as your previous script (iloc[:, 5] = close)

# ============================================================
# === LOAD ===================================================
# ============================================================

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

close = df.iloc[:, CLOSE_COL_IDX].to_numpy(dtype=float)
log_ret = np.diff(np.log(close))
abs_ret = np.abs(log_ret)

print(f"n candles            : {len(close)}")
print(f"n log-returns        : {len(log_ret)}")
print(f"mean log-return      : {log_ret.mean():.6e}")
print(f"std log-return       : {log_ret.std():.6e}")
print(f"mean |log-return|    : {abs_ret.mean():.6e}")

# ============================================================
# === ACF ====================================================
# ============================================================

def acf(x, max_lag):
    """Sample autocorrelation function up to max_lag (lag 0 included)."""
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = np.dot(x[:-k], x[k:]) / (n * var)
    return out

acf_ret = acf(log_ret, MAX_LAG)
acf_abs = acf(abs_ret, MAX_LAG)

# 95% confidence band under white-noise null
ci = 1.96 / np.sqrt(len(log_ret))

# ============================================================
# === FIND DECAY LANDMARKS ===================================
# ============================================================

def first_crossing(a, threshold):
    """First lag k >= 1 where a[k] drops below threshold. Returns None if never."""
    below = np.where(a[1:] < threshold)[0]
    return int(below[0] + 1) if len(below) else None

# vol clustering decay landmarks
lag_abs_1_over_e = first_crossing(acf_abs, acf_abs[0] / np.e)  # ACF(|r|) at lag 0 is 1, so this is just 1/e
lag_abs_005      = first_crossing(acf_abs, 0.05)
lag_abs_ci       = first_crossing(acf_abs, ci)

# momentum decay landmark
lag_ret_ci = first_crossing(np.abs(acf_ret), ci)  # first lag where |ACF(r)| stays below CI

print("\n--- Vol clustering (ACF of |log_return|) ---")
print(f"first lag where ACF < 1/e ({1/np.e:.3f}) : {lag_abs_1_over_e}")
print(f"first lag where ACF < 0.05              : {lag_abs_005}")
print(f"first lag where ACF < 95% CI ({ci:.4f})  : {lag_abs_ci}")

print("\n--- Momentum (ACF of log_return) ---")
print(f"first lag where |ACF| < 95% CI          : {lag_ret_ci}")

# ============================================================
# === PLOT ===================================================
# ============================================================

lags = np.arange(MAX_LAG + 1)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

# --- |returns| ACF (vol clustering) ---
axes[0].bar(lags, acf_abs, width=1.0, color="steelblue", alpha=0.7)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axhline( ci, color="red", linestyle="--", linewidth=0.8, label=f"95% CI = ±{ci:.4f}")
axes[0].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
axes[0].axhline(0.05, color="orange", linestyle=":", linewidth=0.8, label="0.05")
if lag_abs_005 is not None:
    axes[0].axvline(lag_abs_005, color="orange", linestyle=":", linewidth=0.8,
                    label=f"ACF<0.05 at lag {lag_abs_005}")
if lag_abs_ci is not None:
    axes[0].axvline(lag_abs_ci, color="red", linestyle=":", linewidth=0.8,
                    label=f"ACF<CI at lag {lag_abs_ci}")
axes[0].set_title("ACF of |log return|  —  vol clustering")
axes[0].set_xlabel("lag (hours)")
axes[0].set_ylabel("autocorrelation")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(alpha=0.3)

# --- returns ACF (momentum) ---
axes[1].bar(lags, acf_ret, width=1.0, color="darkgreen", alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].axhline( ci, color="red", linestyle="--", linewidth=0.8, label=f"95% CI = ±{ci:.4f}")
axes[1].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
axes[1].set_title("ACF of log return  —  momentum / mean reversion")
axes[1].set_xlabel("lag (hours)")
axes[1].set_ylabel("autocorrelation")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()