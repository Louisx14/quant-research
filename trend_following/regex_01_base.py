# in this file we might start actually working onnstrategy research and testing.

import re
import time
import math
import sqlite3
import pyautogui
import pandas as pd
import numpy as np
from pathlib import Path
from selenium import webdriver
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver 
from selenium.webdriver.common.action_chains import ActionChains


URL = "https://www.binance.com/en/trade/SOL_USDC"
DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
PARAMETERS = [15, 25, 34, 43, 52, 61, 70, 75, 80]
OFFSET = 100
LENGHT = 2000
BIAS_LENGHT = 400

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


def regexdf(df, params):
    close_prices = df.iloc[:, 5]

    for window in params:
        x = np.arange(window)

        df[f"reg_{window}"] = (
            close_prices
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

def add_bias(df, lnght = BIAS_LENGHT, weighted=True) :
    close_prices  = df.iloc[:, 5]
    volumes = df.iloc[:, 6]
    
    x = np.arange(lnght)

    def weighted_slope(y):
        w = volumes.loc[y.index].values
        return np.polyfit(x, y.values, 1, w=w)[0]

    if weighted == True :
        df[f"bias_{lnght}"] = (
            close_prices
            .rolling(lnght)
            .apply(weighted_slope, raw=False)
            .round(8)) 
    else :
        df[f"bias_{lnght}"] = (
            close_prices
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

def wnw_diff_spread(df) :



    df['spread_diff'] = df['sig']/df['wsig']
    return df

#df = regexdf(df, PARAMETERS)
df = wregexdf(df, PARAMETERS)
df = add_ave_column(
    df,
    params = PARAMETERS,
    factor = 70,
    weighted = True,
    mode = 'xp',
    colname = 'wsig'
    )
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

df = wnw_diff_spread(df)

plt.hist(df['spread_diff'], bins=50, range=(-1.5, 3))
plt.show()

print(df)

#plot_ave_func(rows)
plotx(df, sh=False)
plot_sigbias_conv(df, mint=120)

plt.show()