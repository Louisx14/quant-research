"""
REGEX on hourly candles — reduced variant with only the bias filter (2026).

Original file: REGEX_005.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path


DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
PARAMETERS = [15, 25, 34, 43, 52, 61, 70, 75, 80]
OFFSET = 100
LENGTH = 10000
BIAS_LENGTH = 400
trade = None
FEE = 0.075
slippage = 0.002
MARGIN_FEE = 0.01
TOLERANCE = 0.1

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
