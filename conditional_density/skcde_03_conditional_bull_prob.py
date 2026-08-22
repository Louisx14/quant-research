import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# === USER INPUTS ============================================
# ============================================================

x_params       = [-0.20, -0.001, 100]   # [x1, x2, steps] -> SL range
p_bull_params  = [0.25, 0.75, 50]       # [x1, x2, steps] -> p_bull range
n              = 100_000              # number of draws (single shared sample table)
FEE            = 0.0015               # 0.15% flat haircut applied once per exit

DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
FORESIGHT = 48
N_BINS = 20                            # bullish quantile bins for conditional sampling
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
# === CONDITIONAL SAMPLER (bullish + bearish only) ===========
# ============================================================

def sample_moves(df_dist, n_rows, n_bins=20, seed=None):
    """
    Draw n_rows (bullish, bearish) pairs from the joint empirical distribution
    via bullish-quantile binning + conditional bearish draw.
    Direction is NOT included here -- generated per p_bull later.
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

    return bullish_samples, bearish_samples, rng


# ============================================================
# === SIMULATION =============================================
# ============================================================

def compute_metrics(direction, bearish, bullish, sl, fee):
    """Vectorized exit-value computation + summary stats for a single (SL, direction) draw."""
    exit_vals = np.where(
        direction == 0,
        np.where(sl > bearish, sl, bullish),
        bullish
    ) - fee

    positive_mask = exit_vals > 0
    n_pos = positive_mask.sum()
    avg_exit          = exit_vals.mean()
    avg_positive_exit = exit_vals[positive_mask].mean() if n_pos else np.nan
    p_positive        = n_pos / len(exit_vals)
    return avg_exit, avg_positive_exit, p_positive


# ============================================================
# === RUN ====================================================
# ============================================================

df_dist = comp_comingvol(foresight=FORESIGHT, df=df_raw)

# 1. One shared sample table for moves (independent of p_bull)
bullish_samples, bearish_samples, rng = sample_moves(
    df_dist, n_rows=n, n_bins=N_BINS, seed=SEED
)

# 2. Build grids
sl_grid     = np.linspace(x_params[0], x_params[1], x_params[2])
pbull_grid  = np.linspace(p_bull_params[0], p_bull_params[1], p_bull_params[2])

# 3. Allocate result matrices [p_bull, SL]
shape = (len(pbull_grid), len(sl_grid))
Z_avg_exit  = np.empty(shape)
Z_avg_pos   = np.empty(shape)
Z_p_pos     = np.empty(shape)

# 4. Sweep
for i, p_bull in enumerate(pbull_grid):
    # Regenerate just the direction column for this p_bull
    direction = rng.binomial(1, p_bull, size=n)
    for j, sl in enumerate(sl_grid):
        ae, ap, pp = compute_metrics(direction, bearish_samples, bullish_samples, sl, FEE)
        Z_avg_exit[i, j] = ae
        Z_avg_pos[i, j]  = ap
        Z_p_pos[i, j]    = pp

# ============================================================
# === PLOT ===================================================
# ============================================================

SL_mesh, P_mesh = np.meshgrid(sl_grid, pbull_grid)

fig = plt.figure(figsize=(18, 6))

surfaces = [
    (Z_avg_exit,  "Average exit (all trades)",  cm.viridis),
    (Z_avg_pos,   "Average exit (winners only)", cm.plasma),
    (Z_p_pos,     "P(exit > 0)",                 cm.magma),
]

for k, (Z, title, cmap) in enumerate(surfaces, start=1):
    ax = fig.add_subplot(1, 3, k, projection="3d")
    surf = ax.plot_surface(
        SL_mesh, P_mesh, Z,
        cmap=cmap, edgecolor="none", alpha=0.92,
    )
    ax.set_xlabel("SL")
    ax.set_ylabel("p_bull")
    ax.set_zlabel(title.split("(")[0].strip())
    ax.set_title(title + f"\n(with {FEE*100:.2f}% fee)")
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    ax.view_init(elev=22, azim=-130)

plt.tight_layout()
plt.show()