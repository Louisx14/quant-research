"""
Expected volatility events (EVE) — how far price travels in a window (Mar 2026).

For each candle, the maximum upward and downward excursion over the next
`param` candles ("bullish / bearish exposition"), the same measure over the
past window, and the relation between past and future excursions and the
REGEX signal. The first step of the SKCDE work in `conditional_density/`:
before modelling a stop/target rule, know the distribution of the moves it
would be exposed to.

Original file: EVE003.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection on older matplotlib)


DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
PARAMETERS = [15, 25, 34, 43, 52, 61, 70, 75, 80]

OFFSET = 100
LENGTH = 1000
BIAS_LENGTH = 400
trade = None
FEE = 0.1
MARGIN_FEE = 0.01

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
# Load directly into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

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

def EVE(df, param) :
    df['bullexposition'] = 0.0
    df['bearexposition'] = 0.0
    bullval, bearval = [], []
    for i in range(len(df) - param) :
        working_slot = df['close'][i:i+param]
        minim, maxim = min(working_slot), max(working_slot)
        df.loc[i, 'bullexposition'] = 100*maxim/working_slot.iloc[0] - 100
        df.loc[i, 'bearexposition'] = 100*working_slot.iloc[0]/minim - 100
    
    df = df[:-param]
        
    return df

def EVE_vol(df, param):
    results = []
    for i in range(len(df) - param):
        workingslot = df['close'][i:i+param]
        current_vol = df['volume'][i]

        minim, maxim = min(workingslot), max(workingslot)
        driftbull = (100 * maxim / workingslot.iloc[0]) - 100
        driftbear = (100 * minim / workingslot.iloc[0]) - 100

        results.append([driftbull, driftbear, current_vol]) 

    driftbulls = [r[0] for r in results]
    driftbears = [r[1] for r in results]
    volumes = [r[2] for r in results]

    plt.scatter(driftbulls, volumes, color='green', s = 1, alpha = 0.2)
    plt.scatter(driftbears, volumes, color='red', s = 1, alpha = 0.2)

    plt.show()

    return results

def EVE_past(df, param, plot_candles=False):
    results = []
    for i in range(param, len(df) - param):
        workingslot = df['close'][i:i+param]
        pastslot = df['close'][i-param:i]

        minim, maxim = min(workingslot), max(workingslot)
        driftbull = (100 * maxim / workingslot.iloc[0]) - 100
        driftbear = (100 * minim / workingslot.iloc[0]) - 100

        minpast, maxpast = min(pastslot), max(pastslot)
        dblp = (100 * maxim / pastslot.iloc[0]) - 100
        dbrp = (100 * minim / pastslot.iloc[0]) - 100
        #print(f"dblp = {dblp} | dbrp = {dbrp} | form not div = {abs(dblp)+abs(dbrp)}")
        form = (abs(dblp)+abs(dbrp))/2

        results.append([driftbull, driftbear, form, dblp, dbrp]) 

    driftbulls = [r[0] for r in results]
    driftbears = [r[1] for r in results]
    volumes = [r[2] for r in results]

    if plot_candles == True :
        plt.scatter(driftbulls, volumes, color='green', s = 1, alpha = 0.2)
        plt.scatter(driftbears, volumes, color='red', s = 1, alpha = 0.2)

        plt.ylabel("predictor (past)")
        plt.xlabel("predicted (future)")

        plt.show()

    return results

def filter_by_threshold(lstres, threshold) :
    kept = []
    for i in range(len(lstres)) :
        if lstres[i][2] > threshold :
            kept.append(lstres[i])
        
    return kept

def scatter_panel(lstres):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    options = ["neg", "pos"]

    for i, past in enumerate(options):
        for j, future in enumerate(options):

            index_past = 4 if past == "neg" else 3
            index_future = 1 if future == "neg" else 0

            pastdt = [x[index_past] for x in lstres]
            futuredt = [x[index_future] for x in lstres]

            ax = axes[i, j]
            ax.scatter(futuredt, pastdt, s=1, alpha=0.2)
            ax.set_title(f"past: {past}, future: {future}")
            ax.set_xlabel("predicted (future)")
            ax.set_ylabel("predictor (past)")

    plt.tight_layout()
    plt.show()

def regex_corr(df, lstres):

    listislist = ['wreg_15', 'wreg_25', 'wreg_34', 'wreg_43', 'wreg_52', 'wreg_61', 'wreg_70', 'wreg_75', 'wreg_80', 'wsig']

    param = 48

    driftbull = [x[0] for x in lstres]
    driftbear = [x[1] for x in lstres]
    dblp = [x[3] for x in lstres]
    dbrp = [x[4] for x in lstres]

    for string in listislist :
        siglst = []
        for i in range(param, len(df) - param):
            siglst.append(df[f'{string}'].iloc[i])   # ← important fix

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(siglst, driftbull, dblp, s=1, alpha=0.2, color="blue")
        ax.scatter(siglst, driftbear, dbrp, s=1, alpha=0.2, color="red")

        ax.set_xlabel("signal")
        ax.set_ylabel("future drift")
        ax.set_zlabel("past drift")

        plt.show()

df = weighted_regex_df(df, PARAMETERS).dropna().reset_index(drop=True)
df = add_ave_column(df, params = PARAMETERS, factor = 70, weighted = True, mode = 'xp', colname = 'wsig')
df = EVE(df, 48)

print(df)

#print(df) 
#b1 = EVE_past(df, 48)
#scatter_panel(b1)
#regex_corr(df, b1)