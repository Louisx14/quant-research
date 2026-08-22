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

TABLES = {
    "BTC":  "Bitcoin1H",
    "SOL":  "Solana1H",
    "USDC": "USDCUSDT1H",
}

WINDOWS = [24, 168, 720]   # 1 day, 1 week, ~1 month

USDC_CLIP_Q = 0.99         # clip USDC volume spikes at 99th percentile
ANNUALIZE   = np.sqrt(24 * 365)

# ============================================================
# === LOAD ===================================================
# ============================================================

def load(table):
    conn = sqlite3.connect(DB_DIR / DATABASE)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

raw = {k: load(t) for k, t in TABLES.items()}
common_idx = raw["BTC"].index.intersection(raw["SOL"].index).intersection(raw["USDC"].index)
common_idx = common_idx.sort_values()
btc, sol, usdc = (raw[k].loc[common_idx] for k in ("BTC", "SOL", "USDC"))

print(f"Overlap window : {common_idx.min()}  ->  {common_idx.max()}")
print(f"N rows         : {len(common_idx)}")

# ============================================================
# === PER-BAR VARIANCES + CLIPPED USDC VOLUME ================
# ============================================================

def per_bar_var(d):
    h = d["high"].to_numpy(dtype=float)
    l = d["low" ].to_numpy(dtype=float)
    o = d["open"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    var_park = (1.0 / (4.0 * np.log(2.0))) * log_hl ** 2
    var_gk   = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return pd.Series(var_park, index=d.index), pd.Series(var_gk, index=d.index)

btc_park_var, btc_gk_var = per_bar_var(btc)
sol_park_var, sol_gk_var = per_bar_var(sol)

usdc_vol = usdc["volume"].astype(float)
clip_thr = float(np.nanquantile(usdc_vol.values, USDC_CLIP_Q))
usdc_vol_clipped = usdc_vol.clip(upper=clip_thr)
print(f"USDC clip thr  : {clip_thr:,.2f}  (q={USDC_CLIP_Q})\n")

# ============================================================
# === ROLLING SERIES =========================================
# ============================================================
# For a window w:
#   - Vol series = sqrt(rolling mean of per-bar variance) * annualize
#   - USDC volume series = rolling mean of clipped raw volume
# Correlations computed on these rolling-smoothed series.

def roll_vol(var_series, w):
    return np.sqrt(var_series.rolling(w, min_periods=w).mean()) * ANNUALIZE

def roll_mean(series, w):
    return series.rolling(w, min_periods=w).mean()

SERIES_ORDER = [
    "BTC_Park",
    "BTC_GK",
    "SOL_Park",
    "SOL_GK",
    "USDC_Vol",
]

def build_panel(w):
    return pd.DataFrame({
        "BTC_Park": roll_vol(btc_park_var, w),
        "BTC_GK":   roll_vol(btc_gk_var,   w),
        "SOL_Park": roll_vol(sol_park_var, w),
        "SOL_GK":   roll_vol(sol_gk_var,   w),
        "USDC_Vol": roll_mean(usdc_vol_clipped, w),
    })[SERIES_ORDER].dropna()

# ============================================================
# === CORRELATION + PLOT =====================================
# ============================================================

fig, axes = plt.subplots(2, len(WINDOWS), figsize=(5.5 * len(WINDOWS), 10))
if len(WINDOWS) == 1:
    axes = axes[:, None]

METHODS = [("pearson", "Pearson"), ("spearman", "Spearman")]

for col, w in enumerate(WINDOWS):
    panel = build_panel(w)
    pearson  = panel.corr(method="pearson")
    spearman = panel.corr(method="spearman")

    print(f"=== window = {w}h   (n = {len(panel)}) ===")
    print("Pearson:")
    print(pearson.round(3).to_string())
    print("Spearman:")
    print(spearman.round(3).to_string())
    print()

    for row, (mat, name) in enumerate([(pearson, "Pearson"), (spearman, "Spearman")]):
        ax = axes[row, col]
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(SERIES_ORDER)))
        ax.set_yticks(range(len(SERIES_ORDER)))
        ax.set_xticklabels(SERIES_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(SERIES_ORDER, fontsize=9)
        ax.set_title(f"{name}  —  window = {w}h  (n={len(panel)})", fontsize=10)

        # annotate cells
        for i in range(len(SERIES_ORDER)):
            for j in range(len(SERIES_ORDER)):
                val = mat.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9)

        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

plt.tight_layout()
plt.show()