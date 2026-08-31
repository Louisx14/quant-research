"""
Random-draw market model: excursions sampled independently from the
empirical bullish / bearish distributions (the naive model the conditional
SKCDE sampling replaced, kept as the baseline).

Original file: random_draw_model_001.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

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


def sample_empirical(df_dist, n_rows, p_bullish=0.5, seed=None):
    """
    Draw n_rows samples where each row has:
      - direction: 1 (bullish) or 0 (bearish), drawn with prob p_bullish
      - bearish_move: sampled from empirical coming_vol_br distribution
      - bullish_move: sampled from empirical coming_vol_bu distribution

    bearish_move and bullish_move are drawn JOINTLY (same row index),
    preserving the correlation observed in the historical data.
    """
    rng = np.random.default_rng(seed)

    br = df_dist["coming_vol_br"].to_numpy()
    bu = df_dist["coming_vol_bu"].to_numpy()

    # Single set of indices -> preserves joint distribution
    idx = rng.integers(0, len(df_dist), size=n_rows)
    bearish_samples = br[idx]
    bullish_samples = bu[idx]

    direction = rng.binomial(1, p_bullish, size=n_rows)

    return pd.DataFrame({
        "direction":    direction,
        "bearish_move": bearish_samples,
        "bullish_move": bullish_samples,
    })


# --- run ---
df_dist = compute_coming_vol(foresight=48, df=df_raw)
sampled = sample_empirical(df_dist, n_rows=1_000_000, seed=42)

print(sampled.head())
print(sampled.describe())
print("\nEmpirical correlation in source data:",
      df_dist[["coming_vol_br", "coming_vol_bu"]].corr().iloc[0, 1])
print("Correlation in sampled data:        ",
      sampled[["bearish_move", "bullish_move"]].corr().iloc[0, 1])