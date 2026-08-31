"""
Rolling annualised volatility of the hourly SOL series (May 2026).

Original file: volatility over time.py
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

WINDOWS = [24, 168, 720]   # 1 day, 1 week, ~1 month  (in hours)

# column indices
TS_COL    = 1
OPEN_COL  = 2
HIGH_COL  = 3
LOW_COL   = 4
CLOSE_COL = 5

# annualization factor for 1H data: sqrt(24 * 365)
ANNUALIZE = np.sqrt(24 * 365)

# ============================================================
# === LOAD ===================================================
# ============================================================

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

ts = pd.to_datetime(df.iloc[:, TS_COL])
o  = df.iloc[:, OPEN_COL ].to_numpy(dtype=float)
h  = df.iloc[:, HIGH_COL ].to_numpy(dtype=float)
l  = df.iloc[:, LOW_COL  ].to_numpy(dtype=float)
c  = df.iloc[:, CLOSE_COL].to_numpy(dtype=float)

# ============================================================
# === PER-BAR VOLATILITY PROXIES =============================
# ============================================================
# Three flavours, each is a per-bar variance estimate:
#   - "std" proxy   : just (log_return)^2          (close-to-close)
#   - Parkinson     : (1 / (4 ln 2)) * log(H/L)^2
#   - Garman-Klass  : 0.5 * log(H/L)^2 - (2 ln 2 - 1) * log(C/O)^2
# A rolling MEAN of these per-bar variances over a window gives variance
# per bar; sqrt() gives per-bar vol; multiply by ANNUALIZE for annualized vol.

log_ret = np.concatenate([[np.nan], np.diff(np.log(c))])
var_std_bar = log_ret ** 2

log_hl = np.log(h / l)
var_park_bar = (1.0 / (4.0 * np.log(2.0))) * log_hl ** 2

log_co = np.log(c / o)
var_gk_bar = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

s_std  = pd.Series(var_std_bar,  index=ts)
s_park = pd.Series(var_park_bar, index=ts)
s_gk   = pd.Series(var_gk_bar,   index=ts)

def rolling_annualized_vol(var_series, window):
    return np.sqrt(var_series.rolling(window, min_periods=window).mean()) * ANNUALIZE

# ============================================================
# === PLOT ===================================================
# ============================================================

fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(14, 4 * len(WINDOWS)),
                         sharex=True)
if len(WINDOWS) == 1:
    axes = [axes]

colors = {"std": "steelblue", "park": "darkorange", "gk": "crimson"}

for ax, w in zip(axes, WINDOWS):
    v_std  = rolling_annualized_vol(s_std,  w)
    v_park = rolling_annualized_vol(s_park, w)
    v_gk   = rolling_annualized_vol(s_gk,   w)

    ax.plot(v_std.index,  v_std.values,  color=colors["std"],  linewidth=0.9, label="Close-to-close std")
    ax.plot(v_park.index, v_park.values, color=colors["park"], linewidth=0.9, label="Parkinson (HL)")
    ax.plot(v_gk.index,   v_gk.values,   color=colors["gk"],   linewidth=0.9, label="Garman-Klass (OHLC)")

    ax.set_ylabel("annualized vol")
    ax.set_title(f"Rolling annualized volatility  —  window = {w}h  "
                 f"({w/24:.0f} day{'s' if w/24 != 1 else ''})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("date")
plt.tight_layout()
plt.show()

# ============================================================
# === SUMMARY ================================================
# ============================================================

print("=== Mean annualized vol over full series ===")
for w in WINDOWS:
    v_std  = rolling_annualized_vol(s_std,  w).mean()
    v_park = rolling_annualized_vol(s_park, w).mean()
    v_gk   = rolling_annualized_vol(s_gk,   w).mean()
    print(f"  window={w:4d}h   std={v_std:.3f}   parkinson={v_park:.3f}   gk={v_gk:.3f}")


print(df.iloc[:5, TS_COL])
print(df.dtypes)