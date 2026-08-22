import re
import time
import math
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
# Fix #8: log-spaced windows so each captures a genuinely different timescale
# (old set had 70/75/80 which were ~0.98 correlated with each other)
PARAMETERS = [8, 16, 32, 64, 128, 256]
OFFSET = 300  # bumped up because longest window is now 256
LENGHT = 10000
BIAS_LENGHT = 400
trade = None
FEE = 0.075
slippage = 0.002
MARGIN_FEE = 0.01
TOLERANCE = 0.1

# Fix #2: cap extreme volume before using it as a regression weight.
# One liquidation candle shouldn't dominate a 64-bar polyfit.
VOLUME_WINSOR_QUANTILE = 0.95

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
# Load directly into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()


def _prep_volume_weights(volumes):
    """
    Fix #2: turn raw volume into a sane regression weight.
      - log1p to tame the heavy right tail
      - winsorize at the 95th percentile so whale bars don't dominate
    Returns a pd.Series aligned with the input index.
    """
    v = volumes.astype(float).clip(lower=0.0)
    log_v = np.log1p(v)
    cap = log_v.quantile(VOLUME_WINSOR_QUANTILE)
    return log_v.clip(upper=cap)


def regexdf(df, params):
    # Fix #1: regress log(close), not close. Slope units become
    # "log-return per bar" -> comparable across price regimes.
    log_close = np.log(df.iloc[:, 5].astype(float))

    for window in params:
        x = np.arange(window)

        df[f"reg_{window}"] = (
            log_close
            .rolling(window)
            .apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
        )

    return df

def plotx(df, sh=False):

    # Extract columns (change names if needed to match your DB schema)
    open_ = df.iloc[:, 2].astype(float)
    high = df.iloc[:, 3].astype(float)
    low = df.iloc[:, 4].astype(float)
    close = df.iloc[:, 5].astype(float)
    volume = df.iloc[:, 6].astype(float)

    x = np.arange(len(df))

    # Candle components
    top_body = np.maximum(open_, close)
    bottom_body = np.minimum(open_, close)

    top_wick = high
    bottom_wick = low

    colors = np.where(close > open_, "green", "red")

    # Plot wicks
    plt.bar(x, top_wick - bottom_wick,
            width=0.1,
            bottom=bottom_wick,
            color=colors)

    # Plot bodies
    plt.bar(x, top_body - bottom_body,
            width=0.8,
            bottom=bottom_body,
            color=colors)

    if sh:
        plt.show()     

def wregexdf(df, params):
    # Fix #1: regress log(close) for price-regime invariance.
    log_close = np.log(df.iloc[:, 5].astype(float))
    # Fix #2: sanitized volume weights (log1p + 95% winsor).
    weights = _prep_volume_weights(df.iloc[:, 6])

    for window in params:
        x = np.arange(window)

        def weighted_slope(y):
            w = weights.loc[y.index].values
            # Guard against an all-zero weight window (would make polyfit blow up).
            if not np.any(w > 0):
                return np.nan
            return np.polyfit(x, y.values, 1, w=w)[0]

        df[f"wreg_{window}"] = (
            log_close
            .rolling(window)
            .apply(weighted_slope, raw=False)
            .round(8)
        )

    return df

def add_ave_column(df, params=PARAMETERS, factor=70, weighted=True, mode='xp', colname="sig"):
    if mode != 'xp':
        raise NotImplementedError("Only mode='xp' is implemented")

    # Prepare regression columns
    if weighted == False:
        reg_cols = [f"reg_{p}" for p in params]
    else :
        reg_cols = [f"wreg_{p}" for p in params]

    # Make sure all columns exist
    for col in reg_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Precompute weights
    weights = np.exp(np.array(params) / factor)

    # Compute weighted average row-wise
    def weighted_avg(row):
        vals = row[reg_cols].values
        if np.any(pd.isna(vals)):
            return np.nan
        return np.sum(vals * weights) / np.sum(weights)

    df[colname] = df.apply(weighted_avg, axis=1)

    return df

def add_bias(df, lnght = BIAS_LENGHT, weighted=True) :
    # Fix #1: log prices here too, for consistency with the signal.
    log_close = np.log(df.iloc[:, 5].astype(float))
    # Fix #2: same sanitized weights used in wregexdf.
    volumes = _prep_volume_weights(df.iloc[:, 6])

    x = np.arange(lnght)

    def weighted_slope(y):
        w = volumes.loc[y.index].values
        if not np.any(w > 0):
            return np.nan
        return np.polyfit(x, y.values, 1, w=w)[0]

    if weighted == True :
        df[f"bias_{lnght}"] = (
            log_close
            .rolling(lnght)
            .apply(weighted_slope, raw=False)
            .round(8)) 
    else :
        df[f"bias_{lnght}"] = (
            log_close
            .rolling(lnght)
            .apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)) 
        
    return df

def plot_sigbias_conv(df, mint = 0, biasl = BIAS_LENGHT) :
    x = [p for p in range(len(df))]
    for i in range(len(df)) :
        if df["sig"][i + OFFSET]*df[f'bias_{BIAS_LENGHT}'][i + OFFSET] > 0 and df['sig'][i + OFFSET] > 0 :
            plt.plot(x[i], mint, 'o', color='green')
        elif df["sig"][i + OFFSET]*df[f'bias_{BIAS_LENGHT}'][i + OFFSET] > 0 and df['sig'][i + OFFSET] < 0 :
            plt.plot(x[i], mint, 'o', color='red')

def setup_df(df, w = True) :
    if w == True :
        df = wregexdf(df, PARAMETERS)
        df = add_ave_column(
            df,
            params = PARAMETERS,
            factor = 128,
            weighted = True,
            mode = 'xp',
            colname = 'wsig'
            )
    else :
        df = regexdf(df, PARAMETERS)
        df = add_ave_column(
            df,
            params = PARAMETERS,
            factor = 70,
            weighted = False,
            mode = 'xp',
            colname = 'sig'
            )
    df = add_bias(df)
    df = df.iloc[OFFSET:OFFSET+LENGHT]

    return df

def reg_I_shouldve_done_long_ago(df, h):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if h <= 0:
        raise ValueError("h must be > 0")

    n = len(df)

    future_max = np.full(n, np.nan)
    future_min = np.full(n, np.nan)

    highs = df["high"].values
    lows = df["low"].values

    # --- compute forward extrema ---
    for i in range(n - h):
        future_max[i] = np.max(highs[i+1:i+1+h])
        future_min[i] = np.min(lows[i+1:i+1+h])

    df["future_max"] = future_max
    df["future_min"] = future_min

    # --- convert to returns ---
    df["future_max_ret"] = (df["future_max"] - df["close"]) / df["close"]
    df["future_min_ret"] = (df["future_min"] - df["close"]) / df["close"]

    # --- clean dataset ---
    reg_df = df.dropna(subset=["wsig", "future_max_ret", "future_min_ret"])

    x = reg_df["wsig"].values
    y_max = reg_df["future_max_ret"].values
    y_min = reg_df["future_min_ret"].values

    # --- linear regression (numpy polyfit) ---
    coef_max = np.polyfit(x, y_max, 1)
    coef_min = np.polyfit(x, y_min, 1)

    slope_max, intercept_max = coef_max
    slope_min, intercept_min = coef_min

    print("=== REGRESSION RESULTS ===")
    print(f"Max Ret: slope={slope_max:.6f}, intercept={intercept_max:.6f}")
    print(f"Min Ret: slope={slope_min:.6f}, intercept={intercept_min:.6f}")

    # --- R² computation ---
    def r2(y, y_pred):
        return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    y_max_pred = slope_max * x + intercept_max
    y_min_pred = slope_min * x + intercept_min

    print(f"Max Ret R²: {r2(y_max, y_max_pred):.4f}")
    print(f"Min Ret R²: {r2(y_min, y_min_pred):.4f}")

    # --- plotting ---
    plt.figure()

    # scatter
    plt.scatter(x, y_min, s=5, alpha=0.3, color="red", label="Future Min (risk)")
    plt.scatter(x, y_max, s=5, alpha=0.3, color="blue", label="Future Max (reward)")

    # regression lines
    x_line = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(x_line, slope_min * x_line + intercept_min, color="red", linewidth=2)
    plt.plot(x_line, slope_max * x_line + intercept_max, color="blue", linewidth=2)

    plt.title(f"Forward {h}-step excursions vs wsig (with regression)")
    plt.xlabel("wsig")
    plt.ylabel("Future returns")

    plt.legend()
    plt.show()

    return df

df = setup_df(df).dropna()
print(df)
reg_I_shouldve_done_long_ago(df, h=48)

plt.hist(df["wsig"], bins=50)
plt.show()