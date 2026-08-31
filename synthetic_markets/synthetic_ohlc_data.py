"""
Block-bootstrap synthetic OHLC series (May 2026).

Stitches random 250-candle blocks of the real log-return sample into a
series 100x longer, then compares moments and quantiles of real vs synthetic
returns (they match to a few %). The journal's own caveat applies: a
strategy evaluated on these paths sees genuinely new data only across block
boundaries, so they are a null benchmark, not a training set.

Original file: synthetic_OHLCdata.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# === USER INPUTS ============================================
# ============================================================

DB_DIR    = Path("data")
DATABASE  = "crypto.db"
TABLENAME = "solana1H"

MULTIPLIER         = 100      # generate MULTIPLIER * len(real) candles
MEAN_BLOCK_LENGTH  = 250      # mean of geometric distribution for stationary bootstrap
SEED               = 42

# column indices in the raw table (matches your previous code)
TS_COL    = 0   # timestamp
OPEN_COL  = 2
HIGH_COL  = 3
LOW_COL   = 4
CLOSE_COL = 5

# panel plot
N_SYNTH_PANELS = 18           # 5x4 grid, last 2 are reference
N_REF_PANELS   = 2
PANEL_CANDLES  = 300

# ============================================================
# === LOAD ===================================================
# ============================================================

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

ts    = pd.to_datetime(df.iloc[:, TS_COL])
o     = df.iloc[:, OPEN_COL].to_numpy(dtype=float)
h     = df.iloc[:, HIGH_COL].to_numpy(dtype=float)
l     = df.iloc[:, LOW_COL].to_numpy(dtype=float)
c     = df.iloc[:, CLOSE_COL].to_numpy(dtype=float)
N_real = len(c)

# hour of week 0..167  (used for stratified bootstrap)
how = (ts.dt.dayofweek * 24 + ts.dt.hour).to_numpy()

# ============================================================
# === EXTRACT PER-CANDLE 4-TUPLE OF LOG-RETURNS =============
# ============================================================
# For each candle t (t>=1) we keep:
#   g_t = log(open_t / close_{t-1})   -> gap return  (carries gap structure)
#   b_t = log(close_t / open_t)        -> body return
#   u_t = log(high_t  / open_t)        -> upper wick (>=0 if open<high, can be 0)
#   d_t = log(low_t   / open_t)        -> lower wick (<=0 if open>low, can be 0)
# These four numbers fully describe candle t's geometry relative to the prior close.
# Bootstrapping rows of this table (with hour-of-week alignment) and compounding
# from a chosen starting price reconstructs a synthetic OHLC series.

gap  = np.log(o[1:] / c[:-1])
body = np.log(c[1:] / o[1:])
up   = np.log(h[1:] / o[1:])
dn   = np.log(l[1:] / o[1:])
how1 = how[1:]                     # hour-of-week aligned to candles 1..N-1

n_tuples = len(gap)
print(f"Real candles               : {N_real}")
print(f"Usable 4-tuples            : {n_tuples}")
print(f"Mean block length          : {MEAN_BLOCK_LENGTH}")
print(f"Synth target candles       : {MULTIPLIER * N_real}")

# ============================================================
# === BUILD HOUR-OF-WEEK INDEX BUCKETS =======================
# ============================================================
# For each h in 0..167, list the row indices (into the 4-tuple table) whose
# hour-of-week is h. Bootstrap will pick a START index from the bucket
# matching the current synthetic hour-of-week, then walk forward
# contiguously (in real-data time) until a geometric block length expires.

buckets = [np.where(how1 == k)[0] for k in range(168)]
empty = [k for k, b in enumerate(buckets) if len(b) == 0]
if empty:
    print(f"WARNING: empty hour-of-week buckets: {empty}")

all_idx = np.arange(n_tuples)

def resolve_bucket(buckets, k):
    """Return a non-empty index pool for hour-of-week k.

    If bucket k is empty (a gap in the real data), walk outward to the
    nearest hour-of-week that has data so weekly seasonality is still
    approximately preserved; only fall back to the full series if every
    bucket is empty.
    """
    if len(buckets[k]):
        return buckets[k]
    for off in range(1, 168):
        for cand in ((k + off) % 168, (k - off) % 168):
            if len(buckets[cand]):
                return buckets[cand]
    return all_idx

# ============================================================
# === STATIONARY BOOTSTRAP (HOUR-OF-WEEK STRATIFIED) =========
# ============================================================
# Standard stationary bootstrap: at each step, with probability 1/L start a new
# block (else continue the current one by stepping forward in the real series).
# Stratified twist: when we START a new block, we draw the seed index from the
# bucket of the current synthetic hour-of-week. This guarantees the synthetic
# series at hour-of-week h is sampled from real data observed at hour-of-week h,
# preserving intraday/weekly seasonality of vol.

def stratified_stationary_bootstrap(n_synth, buckets, n_tuples, mean_L, seed):
    rng = np.random.default_rng(seed)
    p_new_block = 1.0 / mean_L
    idx = np.empty(n_synth, dtype=np.int64)

    # start the synthetic series at synthetic-hour-of-week 0
    # (arbitrary; doesn't matter, the series is stationary)
    cur_how = 0
    bucket = resolve_bucket(buckets, cur_how)
    cur = int(rng.choice(bucket))
    idx[0] = cur

    # precompute random draws for speed
    new_block_draws = rng.random(n_synth)

    for t in range(1, n_synth):
        # advance synthetic hour-of-week (each synthetic candle = 1 hour)
        cur_how = (cur_how + 1) % 168

        if new_block_draws[t] < p_new_block:
            # start new block: draw from bucket of current synthetic h-o-w
            bucket = resolve_bucket(buckets, cur_how)
            cur = int(rng.choice(bucket))
        else:
            # continue current block: step forward in real series, wrap if needed
            cur = (cur + 1) % n_tuples
            # if walking forward put us at a row whose real h-o-w doesn't match
            # synthetic h-o-w, that's a tolerated phase slip WITHIN a block
            # (within-block we prioritize contiguity over alignment, since
            # contiguity is what carries the clustering signal). Re-alignment
            # happens automatically at the next block start.
        idx[t] = cur
    return idx

n_synth = MULTIPLIER * N_real
print("Bootstrapping...")
idx = stratified_stationary_bootstrap(n_synth, buckets, n_tuples, MEAN_BLOCK_LENGTH, SEED)

# ============================================================
# === RECONSTRUCT SYNTHETIC OHLC =============================
# ============================================================

g_s = gap[idx]
b_s = body[idx]
u_s = up[idx]
d_s = dn[idx]

# Reconstruct prices by chaining: each candle's open = prior close * exp(gap)
#                                  each candle's close = open * exp(body)
# Then high = open * exp(upper wick), low = open * exp(lower wick).
start_price = float(c[0])
log_open    = np.empty(n_synth)
log_close   = np.empty(n_synth)

# first candle: anchor on start_price as the "previous close"
log_open[0]  = np.log(start_price) + g_s[0]
log_close[0] = log_open[0] + b_s[0]
for t in range(1, n_synth):
    log_open[t]  = log_close[t-1] + g_s[t]
    log_close[t] = log_open[t]    + b_s[t]

# vectorize wicks
synth_open  = np.exp(log_open)
synth_close = np.exp(log_close)
synth_high  = synth_open * np.exp(u_s)
synth_low   = synth_open * np.exp(d_s)

# sanity: high >= max(open,close), low <= min(open,close) by construction since
# u_s = log(h/o) >= log(max(o,c)/o) (true in real data => true after copy),
# d_s = log(l/o) <= log(min(o,c)/o) likewise.
ok_h = (synth_high >= np.maximum(synth_open, synth_close) - 1e-12).all()
ok_l = (synth_low  <= np.minimum(synth_open, synth_close) + 1e-12).all()
print(f"OHLC consistency check (high)  : {ok_h}")
print(f"OHLC consistency check (low)   : {ok_l}")

synth = pd.DataFrame({
    "open":  synth_open,
    "high":  synth_high,
    "low":   synth_low,
    "close": synth_close,
})

# ============================================================
# === STATS COMPARISON =======================================
# ============================================================

def summarize(series_close):
    lr = np.diff(np.log(series_close))
    return {
        "n_candles":      len(series_close),
        "mean_log_ret":   lr.mean(),
        "std_log_ret":    lr.std(),
        "mean_abs_ret":   np.abs(lr).mean(),
        "skew":           pd.Series(lr).skew(),
        "kurtosis":       pd.Series(lr).kurt(),
        "q01":            np.quantile(lr, 0.01),
        "q05":            np.quantile(lr, 0.05),
        "q50":            np.quantile(lr, 0.50),
        "q95":            np.quantile(lr, 0.95),
        "q99":            np.quantile(lr, 0.99),
    }

real_stats  = summarize(c)
synth_stats = summarize(synth["close"].to_numpy())

stats_df = pd.DataFrame({"real": real_stats, "synth": synth_stats})
stats_df["abs_diff"] = (stats_df["synth"] - stats_df["real"]).abs()
stats_df["rel_diff"] = stats_df["abs_diff"] / stats_df["real"].abs()

print("\n=== Stats comparison ===")
print(stats_df.to_string(float_format=lambda x: f"{x:.6e}"))

# ============================================================
# === PANEL PLOT: 5x4 candlesticks ===========================
# ============================================================

def draw_candles(ax, o_, h_, l_, c_, title):
    n = len(o_)
    width = 0.7
    for i in range(n):
        color = "#26a69a" if c_[i] >= o_[i] else "#ef5350"
        # wick
        ax.plot([i, i], [l_[i], h_[i]], color=color, linewidth=0.5)
        # body
        body_lo = min(o_[i], c_[i])
        body_hi = max(o_[i], c_[i])
        height = max(body_hi - body_lo, (h_[i] - l_[i]) * 1e-3)  # avoid 0-height
        ax.add_patch(Rectangle((i - width/2, body_lo), width, height,
                               facecolor=color, edgecolor=color, linewidth=0.3))
    ax.set_xlim(-1, n)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)

rng_plot = np.random.default_rng(SEED + 1)

fig, axes = plt.subplots(5, 4, figsize=(20, 14))
axes = axes.flatten()

# 18 synthetic panels
for i in range(N_SYNTH_PANELS):
    start = int(rng_plot.integers(0, n_synth - PANEL_CANDLES))
    sl = slice(start, start + PANEL_CANDLES)
    draw_candles(axes[i],
                 synth["open"].to_numpy()[sl],
                 synth["high"].to_numpy()[sl],
                 synth["low"].to_numpy()[sl],
                 synth["close"].to_numpy()[sl],
                 f"SYNTH  [{start}:{start+PANEL_CANDLES}]")

# 2 reference panels at the end
for j in range(N_REF_PANELS):
    start = int(rng_plot.integers(0, N_real - PANEL_CANDLES))
    sl = slice(start, start + PANEL_CANDLES)
    ax = axes[N_SYNTH_PANELS + j]
    draw_candles(ax, o[sl], h[sl], l[sl], c[sl],
                 f"REAL   [{start}:{start+PANEL_CANDLES}]")
    # outline real panels in black so they stand out
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

plt.suptitle(
    f"Synthetic vs Real OHLC  |  mean block = {MEAN_BLOCK_LENGTH}  |  "
    f"synth std/real std = {synth_stats['std_log_ret']/real_stats['std_log_ret']:.3f}",
    fontsize=12, y=0.995
)
plt.tight_layout()
plt.show()