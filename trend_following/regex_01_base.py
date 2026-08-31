"""
REGEX on hourly candles — base version (Feb 2026).

Adds one OLS slope column per lookback in PARAMETERS to the candle frame
(`regex_df`), a volume-weighted variant (`weighted_regex_df`), the
exp(n/70)-weighted signal (`add_ave_column`), a long-lookback "bias"
regression (`add_bias`) and plots the signal line coloured by whether signal
and bias agree. `weighted_vs_unweighted_spread` produces the histogram of
weighted / unweighted signal ratios discussed in the journal (18 Feb 2026).

Original file: REGEX_001.py (a copy of WS_009.py)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


URL = "https://www.binance.com/en/trade/SOL_USDC"
DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
PARAMETERS = [15, 25, 34, 43, 52, 61, 70, 75, 80]
OFFSET = 100
LENGTH = 2000
BIAS_LENGTH = 400

collst = [
    "#ff0000",
    "#ff3366",
    "#ff66cc",
    "#cc66ff",
    "#9933ff",
    "#6633cc",
    "#4d4dff",
    "#3333cc",
    "#0000ff",
]

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)

# Load directly into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()


def regex_df(df, params):
    close_prices = df.iloc[:, 5]

    for window in params:
        x = np.arange(window)

        df[f"reg_{window}"] = (
            close_prices
            .rolling(window)
            .apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
        )

    return df

def plot_candles(df, sh=False):

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

def weighted_regex_df(df, params):
    close_prices = df.iloc[:, 5].astype(float)
    weights = df.iloc[:, 6].astype(float)

    for window in params:
        x = np.arange(window)

        def weighted_slope(y):
            w = weights.loc[y.index].values
            return np.polyfit(x, y.values, 1, w=w)[0]

        df[f"wreg_{window}"] = (
            close_prices
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

def add_bias(df, length = BIAS_LENGTH, weighted=True) :
    close_prices  = df.iloc[:, 5]
    volumes = df.iloc[:, 6]
    
    x = np.arange(length)

    def weighted_slope(y):
        w = volumes.loc[y.index].values
        return np.polyfit(x, y.values, 1, w=w)[0]

    if weighted == True :
        df[f"bias_{length}"] = (
            close_prices
            .rolling(length)
            .apply(weighted_slope, raw=False)
            .round(8)) 
    else :
        df[f"bias_{length}"] = (
            close_prices
            .rolling(length)
            .apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)) 
        
    return df

def plot_sigbias_conv(df, start_index = 0, bias_length = BIAS_LENGTH) :
    x = [p for p in range(len(df))]
    for i in range(len(df)) :
        if df["sig"][i + OFFSET]*df[f'bias_{BIAS_LENGTH}'][i + OFFSET] > 0 and df['sig'][i + OFFSET] > 0 :
            plt.plot(x[i], start_index, 'o', color='green')
        elif df["sig"][i + OFFSET]*df[f'bias_{BIAS_LENGTH}'][i + OFFSET] > 0 and df['sig'][i + OFFSET] < 0 :
            plt.plot(x[i], start_index, 'o', color='red')

def weighted_vs_unweighted_spread(df) :



    df['spread_diff'] = df['sig']/df['wsig']
    return df

#df = regex_df(df, PARAMETERS)
df = weighted_regex_df(df, PARAMETERS)
df = add_ave_column(
    df,
    params = PARAMETERS,
    factor = 70,
    weighted = True,
    mode = 'xp',
    colname = 'wsig'
    )
df = regex_df(df, PARAMETERS)
df = add_ave_column(
    df,
    params = PARAMETERS,
    factor = 70,
    weighted = False,
    mode = 'xp',
    colname = 'sig'
    )
df = add_bias(df)
df = df.iloc[OFFSET:OFFSET+LENGTH]

df = weighted_vs_unweighted_spread(df)

plt.hist(df['spread_diff'], bins=50, range=(-1.5, 3))
plt.show()

print(df)

#plot_ave_func(rows)
plot_candles(df, sh=False)
plot_sigbias_conv(df, start_index=120)

plt.show()