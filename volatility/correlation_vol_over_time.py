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

WINDOWS = [24, 168, 720]   # 1 day, 1 week, ~1 month  (in hours)

# annualization factor for 1H data: sqrt(24 * 365)
ANNUALIZE = np.sqrt(24 * 365)

# colors  (tuned for dark background)
COLORS = {
    "BTC_park": "#B22222",   # bordeaux / dark red
    "BTC_gk":   "#FF4444",   # bright red
    "SOL_park": "#3A6FCC",   # mid blue
    "SOL_gk":   "#5FB4FF",   # bright sky blue
    "USDC_raw": "#FFEB3B",   # bright yellow (raw, low alpha)
    "USDC_sm":  "#FFC107",   # gold (smoothed)
}
BG = "#0e0e10"
FG = "#e6e6e6"

# clip USDC raw volume to this quantile to avoid a few spikes destroying the scaling
USDC_CLIP_Q = 0.99

# ============================================================
# === LOAD ===================================================
# ============================================================

def load(table):
    conn = sqlite3.connect(DB_DIR / DATABASE)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    # columns: position(0), date(1), open(2), high(3), low(4), close(5), volume(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

raw = {k: load(t) for k, t in TABLES.items()}
for k, d in raw.items():
    print(f"{k:5s}: {len(d):6d} rows, {d.index.min()}  ->  {d.index.max()}")

# ============================================================
# === SYNC: intersection of all three indices ================
# ============================================================

common_idx = raw["BTC"].index.intersection(raw["SOL"].index).intersection(raw["USDC"].index)
common_idx = common_idx.sort_values()
print(f"\nOverlap: {len(common_idx)} rows  ({common_idx.min()} -> {common_idx.max()})")

btc  = raw["BTC"].loc[common_idx]
sol  = raw["SOL"].loc[common_idx]
usdc = raw["USDC"].loc[common_idx]

# ============================================================
# === PER-BAR VOL PROXIES (Parkinson, Garman-Klass) ==========
# ============================================================

def per_bar_var(df):
    h = df["high"].to_numpy(dtype=float)
    l = df["low" ].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    var_park = (1.0 / (4.0 * np.log(2.0))) * log_hl ** 2
    var_gk   = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return pd.Series(var_park, index=df.index), pd.Series(var_gk, index=df.index)

btc_park, btc_gk = per_bar_var(btc)
sol_park, sol_gk = per_bar_var(sol)

usdc_vol_raw = usdc["volume"].astype(float)

def roll_ann_vol(var_series, w):
    return np.sqrt(var_series.rolling(w, min_periods=w).mean()) * ANNUALIZE

# ============================================================
# === PLOT ===================================================
# ============================================================

# clip USDC volume at the 99th percentile (spikes flatten at top of band)
usdc_clip_threshold = float(np.nanquantile(usdc_vol_raw.values, USDC_CLIP_Q))
n_clipped = int((usdc_vol_raw > usdc_clip_threshold).sum())
print(f"\nUSDC volume clipping: q={USDC_CLIP_Q} -> threshold={usdc_clip_threshold:,.2f}")
print(f"  clipped {n_clipped} / {len(usdc_vol_raw)} values "
      f"({100*n_clipped/len(usdc_vol_raw):.2f}%)")
usdc_vol_clipped = usdc_vol_raw.clip(upper=usdc_clip_threshold)

# ----- dark styling -----
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   FG,
    "axes.labelcolor":  FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "text.color":       FG,
    "axes.titlecolor":  FG,
    "grid.color":       "#3a3a3a",
    "legend.facecolor": "#1a1a1c",
    "legend.edgecolor": "#3a3a3a",
})

fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(15, 4.5 * len(WINDOWS)),
                         sharex=True)
if len(WINDOWS) == 1:
    axes = [axes]

for ax, w in zip(axes, WINDOWS):
    # --- vol lines ---
    btc_park_w = roll_ann_vol(btc_park, w)
    btc_gk_w   = roll_ann_vol(btc_gk,   w)
    sol_park_w = roll_ann_vol(sol_park, w)
    sol_gk_w   = roll_ann_vol(sol_gk,   w)

    ax.plot(btc_park_w.index, btc_park_w.values, color=COLORS["BTC_park"],
            linewidth=1.0, label="BTC Parkinson")
    ax.plot(btc_gk_w.index,   btc_gk_w.values,   color=COLORS["BTC_gk"],
            linewidth=1.0, label="BTC Garman-Klass")
    ax.plot(sol_park_w.index, sol_park_w.values, color=COLORS["SOL_park"],
            linewidth=1.0, label="SOL Parkinson")
    ax.plot(sol_gk_w.index,   sol_gk_w.values,   color=COLORS["SOL_gk"],
            linewidth=1.0, label="SOL Garman-Klass")

    # --- compute vol range across BTC+SOL on this panel ---
    all_vol = pd.concat([btc_park_w, btc_gk_w, sol_park_w, sol_gk_w])
    vmin = float(np.nanmin(all_vol.values))
    vmax = float(np.nanmax(all_vol.values))

    # --- USDC volume: raw clipped (low alpha) + smoothed (alpha 1) ---
    usdc_raw = usdc_vol_clipped
    usdc_sm  = usdc_vol_clipped.rolling(w, min_periods=w).mean()

    # rescale BOTH using the SAME (min, max) — derived from the clipped raw —
    # so the smoothed line is a smoothed version of what you see in the raw.
    raw_min = float(np.nanmin(usdc_raw.values))
    raw_max = float(np.nanmax(usdc_raw.values))
    usdc_raw_scaled = (usdc_raw - raw_min) / (raw_max - raw_min) * (vmax - vmin) + vmin
    usdc_sm_scaled  = (usdc_sm  - raw_min) / (raw_max - raw_min) * (vmax - vmin) + vmin

    ax.plot(usdc_raw_scaled.index, usdc_raw_scaled.values, color=COLORS["USDC_raw"],
            linewidth=0.6, alpha=0.30, label="USDC vol (raw, clipped & scaled)")
    ax.plot(usdc_sm_scaled.index, usdc_sm_scaled.values, color=COLORS["USDC_sm"],
            linewidth=1.4, alpha=1.0, label=f"USDC vol (smoothed {w}h)")

    ax.set_ylabel("annualized vol  /  scaled USDC volume")
    ax.set_title(f"Rolling annualized volatility  —  window = {w}h  "
                 f"({w/24:.0f} day{'s' if w/24 != 1 else ''})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

axes[-1].set_xlabel("date")
plt.tight_layout()
plt.show()