"""
Empirical SL/TP grid test on the real sample (Jun 2026).

Replays a stop-loss / take-profit grid directly on the candle table rather
than on SKCDE draws, tracking the realised path, so the conditional-sampling
results can be compared with what the actual sample pays.

RUNTIME: replays the full grid on every candle — several minutes on the full
hourly table.

Original file: empirical_test_across_vars.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# ============================================================
# === USER INPUTS ============================================
# ============================================================

DB_DIR    = Path("data")
DATABASE  = "crypto.db"
TABLENAME = "solana1H"

FORESIGHT = 48

# SL grid (negative returns)  [start, stop, n_steps]
sl_params = [-0.005, -0.000, 50]
# TP grid (positive returns)
tp_params = [ 0.002,  0.035, 50]

TRAIN_FRAC = 0.90
SEED       = 42

FEE = 0.0015   # used only when fees=True is passed to simulate()

# --- Portfolio equity-curve settings (see plot_portfolio_paths) ---
PP_N      = 50       # how many portfolio curves to draw
PP_BEST   = False    # draw the n best performers (printed too)
PP_RAND   = False   # draw n randomly chosen portfolios (seeded by SEED)
PP_SPREAD = True   # draw n portfolios evenly spaced across the cum_pnl ranking
PP_FEES   = False   # apply 2*FEE per trade when building the curves

# column indices (matches previous code: ts=0, open=2, high=3, low=4, close=5)
TS_COL    = 0
OPEN_COL  = 2
HIGH_COL  = 3
LOW_COL   = 4
CLOSE_COL = 5

# ============================================================
# === LOAD ===================================================
# ============================================================

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

o = df.iloc[:, OPEN_COL ].to_numpy(dtype=float)
h = df.iloc[:, HIGH_COL ].to_numpy(dtype=float)
l = df.iloc[:, LOW_COL  ].to_numpy(dtype=float)
c = df.iloc[:, CLOSE_COL].to_numpy(dtype=float)
N = len(c)

# ============================================================
# === 90/10 SPLIT ============================================
# ============================================================
# Chronological split: first 90% is train (used here for the sweep),
# last 10% is held out and untouched.

split = int(np.floor(TRAIN_FRAC * N))
o_tr, h_tr, l_tr, c_tr = o[:split], h[:split], l[:split], c[:split]
o_te, h_te, l_te, c_te = o[split:], h[split:], l[split:], c[split:]

print(f"Total candles  : {N}")
print(f"Train candles  : {len(c_tr)}  (indices 0..{split-1})")
print(f"Test candles   : {len(c_te)}  (held out, untouched)")
print(f"Foresight      : {FORESIGHT}")

# ============================================================
# === SINGLE-TRADE SIMULATOR =================================
# ============================================================
# For one entry at index n (entering long at close[n]), simulate the trade
# over up to `foresight` subsequent candles with the SL/TP/trailing logic.
#
# Logic per candle k = n+1, n+2, ...:
#   Phase A (pre-TP-pierce):
#     1) if low[k]  <= SL_price -> exit at SL_price        (pessimistic ordering)
#     2) elif high[k] >= TP_price:
#          - if close[k] < TP_price -> enter trailing mode (starting k+1)
#          - else                     -> exit at TP_price
#     3) else continue
#   Phase B (trailing mode), starting on candle k_trail+1 = k_pierce+1:
#     for each candle j:
#       cand = close[j-1]
#       if cand > SL_price: SL_price <- cand
#       if low[j] <= SL_price: exit at SL_price
#
# If foresight runs out with no exit: exit at the last available close in window.
# Uses whatever foresight remains for entries near the end of the series.

def simulate_single(o_, h_, l_, c_, n, sl_ret, tp_ret, foresight):
    entry = c_[n]
    sl_price = entry * (1.0 + sl_ret)
    tp_price = entry * (1.0 + tp_ret)

    last_k = min(n + foresight, len(c_) - 1)
    if last_k <= n:
        return 0.0  # nothing to simulate (last candle)

    # --- Phase A ---
    pierced_at = -1
    exit_k = -1
    exit_price = np.nan
    for k in range(n + 1, last_k + 1):
        if l_[k] <= sl_price:
            exit_k = k
            exit_price = sl_price
            break
        if h_[k] >= tp_price:
            if c_[k] < tp_price:
                pierced_at = k
                break
            else:
                exit_k = k
                exit_price = tp_price
                break

    # --- Phase B (trailing) ---
    if exit_k < 0 and pierced_at >= 0:
        for j in range(pierced_at + 1, last_k + 1):
            cand = c_[j - 1]
            if cand > sl_price:
                sl_price = cand
            if l_[j] <= sl_price:
                exit_k = j
                exit_price = sl_price
                break

    # --- Timed out: exit at last available close ---
    if exit_k < 0:
        exit_k = last_k
        exit_price = c_[last_k]

    pnl = exit_price / entry - 1.0
    return pnl


# ============================================================
# === GRID SWEEP =============================================
# ============================================================

def simulate_grid(o_, h_, l_, c_, sl_grid, tp_grid, foresight, fees=False):
    """
    Returns:
      avg_pnl : (len(sl_grid), len(tp_grid)) mean per-trade PnL
      cum_pnl : (len(sl_grid), len(tp_grid)) total compounded growth (prod(1+pnl) - 1)
    Rows index SL values, cols index TP values.
    """
    n_entries = len(c_) - 1   # need at least 1 future candle
    avg = np.empty((len(sl_grid), len(tp_grid)))
    cum = np.empty((len(sl_grid), len(tp_grid)))

    fee_per_trade = (2 * FEE) if fees else 0.0  # entry + exit haircut

    for i, sl in enumerate(sl_grid):
        for j, tp in enumerate(tp_grid):
            pnls = np.empty(n_entries)
            for n in range(n_entries):
                pnls[n] = simulate_single(o_, h_, l_, c_, n, sl, tp, foresight)
            if fees:
                pnls = pnls - fee_per_trade
            avg[i, j] = pnls.mean()
            cum[i, j] = np.prod(1.0 + pnls) - 1.0
        print(f"  SL row {i+1}/{len(sl_grid)} done (sl={sl:+.4f})")

    return avg, cum


sl_grid = np.linspace(sl_params[0], sl_params[1], sl_params[2])
tp_grid = np.linspace(tp_params[0], tp_params[1], tp_params[2])

print("\nRunning grid sweep on training set...")
avg_pnl, cum_pnl = simulate_grid(o_tr, h_tr, l_tr, c_tr, sl_grid, tp_grid,
                                 FORESIGHT, fees=False)

# --- best cells ---
i_best, j_best = np.unravel_index(np.argmax(avg_pnl), avg_pnl.shape)
print(f"\nBest avg PnL : {avg_pnl[i_best, j_best]:+.5f}  "
      f"at SL={sl_grid[i_best]:+.4f}, TP={tp_grid[j_best]:+.4f}")

i_bc, j_bc = np.unravel_index(np.argmax(cum_pnl), cum_pnl.shape)
print(f"Best cum PnL : {cum_pnl[i_bc, j_bc]:+.4e}  "
      f"at SL={sl_grid[i_bc]:+.4f}, TP={tp_grid[j_bc]:+.4f}")

# ============================================================
# === PLOT: 3D surfaces (viridis) ============================
# ============================================================

SL_M, TP_M = np.meshgrid(sl_grid, tp_grid, indexing="ij")

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf1 = ax1.plot_surface(SL_M, TP_M, avg_pnl, cmap="viridis", edgecolor="none")
ax1.set_xlabel("SL")
ax1.set_ylabel("TP")
ax1.set_zlabel("avg PnL")
ax1.set_title(f"Average PnL per trade  (foresight={FORESIGHT})")
fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf2 = ax2.plot_surface(SL_M, TP_M, cum_pnl, cmap="viridis", edgecolor="none")
ax2.set_xlabel("SL")
ax2.set_ylabel("TP")
ax2.set_zlabel("total compounded growth")
ax2.set_title(f"Compounded growth across all trades  (foresight={FORESIGHT})")
fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)

plt.tight_layout()
plt.show()

print(f"avg_pnl shape: {avg_pnl.shape}, cum_pnl shape: {cum_pnl.shape}")
print(f"max avg_pnl: {avg_pnl.max():+.5f} at SL={sl_grid[i_best]:+.4f}, TP={tp_grid[j_best]:+.4f}")
print(f"max cum_pnl: {cum_pnl.max():+.4e} at SL={sl_grid[i_bc]:+.4f}, TP={tp_grid[j_bc]:+.4f}")

# ============================================================
# === PORTFOLIO EQUITY CURVES ================================
# ============================================================
# Each (SL, TP) cell defines a portfolio: enter every candle, compound the
# per-trade PnL. cum_pnl already holds the *final* value of each path; here we
# rebuild the full path through time for a chosen subset of cells and plot them.


def _trade_pnls(o_, h_, l_, c_, sl_ret, tp_ret, foresight, fees=False):
    """Per-trade PnL series for one (SL, TP) cell (same logic as simulate_grid)."""
    n_entries = len(c_) - 1
    pnls = np.empty(n_entries)
    for k in range(n_entries):
        pnls[k] = simulate_single(o_, h_, l_, c_, k, sl_ret, tp_ret, foresight)
    if fees:
        pnls = pnls - 2 * FEE
    return pnls


def plot_portfolio_paths(o_, h_, l_, c_, sl_grid, tp_grid, cum_pnl, foresight,
                         n=10, best=True, rand=False, spread=False, fees=False):
    """
    Draw n portfolio equity curves (equity starts at 1.0, compounds each trade).

    Selection modes are mutually exclusive, priority best > rand > spread:
      best  : the n cells with the highest final cum_pnl (also printed)
      rand  : n randomly chosen cells (uses SEED for reproducibility)
      spread: n cells evenly spaced across the cum_pnl performance ranking
              (worst -> best), to show the full spread of outcomes
    If none is True, falls back to spread.
    """
    n_sl, n_tp = cum_pnl.shape
    cum_flat = cum_pnl.ravel()
    total = cum_flat.size
    n = max(1, min(n, total))

    # ranking of flat indices, worst -> best
    order = np.argsort(cum_flat)

    if best:
        sel = order[::-1][:n]                       # top n, best first
        mode = "best"
    elif rand:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(total, size=n, replace=False)
        mode = "rand"
    else:                                           # spread (also the default)
        pos = np.linspace(0, total - 1, n).round().astype(int)
        sel = order[pos]                            # evenly spaced in ranking
        mode = "spread"

    print(f"\nPortfolio paths: drawing {n} curve(s)  (mode={mode}, fees={fees})")

    plt.figure(figsize=(13, 7))
    for rank, flat in enumerate(sel):
        i, j = np.unravel_index(flat, (n_sl, n_tp))
        sl, tp = sl_grid[i], tp_grid[j]
        pnls = _trade_pnls(o_, h_, l_, c_, sl, tp, foresight, fees=fees)
        equity = np.empty(pnls.size + 1)
        equity[0] = 1.0
        equity[1:] = np.cumprod(1.0 + pnls)
        final = equity[-1] - 1.0
        label = f"SL={sl:+.4f} TP={tp:+.4f}  cum={final:+.3e}"
        if best:
            print(f"  #{rank+1:<3d} {label}")
        plt.plot(np.arange(equity.size), equity, lw=1.0, label=label)

    plt.axhline(1.0, color="k", lw=0.7, ls="--")
    plt.xlabel("trade number")
    plt.ylabel("portfolio equity (start = 1.0)")
    plt.yscale("log")
    plt.title(f"Portfolio equity curves  (mode={mode}, n={n}, foresight={foresight})")
    if n <= 20:
        plt.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.show()


# Driven by the PORTFOLIO EQUITY-CURVE SETTINGS near USER INPUTS.
plot_portfolio_paths(o_tr, h_tr, l_tr, c_tr, sl_grid, tp_grid, cum_pnl,
                     FORESIGHT, n=PP_N, best=PP_BEST, rand=PP_RAND,
                     spread=PP_SPREAD, fees=PP_FEES)