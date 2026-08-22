"""import re
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

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

print(df)

def comp_comingvol(foresight = 48, df = df, colname="coming_vol"):
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

df = comp_comingvol()

print(df)
    


def plot_coming_vol_histograms(df, bins=100):
    fig, ax = plt.subplots(figsize=(12, 6))

    br_mean = df['coming_vol_br'].mean()
    bu_mean = df['coming_vol_bu'].mean()

    # Bearish (close-to-close) — solid red
    ax.hist(df['coming_vol_br'].dropna(), bins=bins,
            color='red', alpha=1, label=f'Bearish (close)  mean={br_mean:.4f}', edgecolor='none')

    # Bullish (close-to-close) — solid green
    ax.hist(df['coming_vol_bu'].dropna(), bins=bins,
            color='green', alpha=1, label=f'Bullish (close)  mean={bu_mean:.4f}', edgecolor='none')

    # Mean lines
    ax.axvline(br_mean, color='darkred', linewidth=1.5, linestyle='--')
    ax.axvline(bu_mean, color='darkgreen', linewidth=1.5, linestyle='--')

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Percentage move over foresight window')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of forward price moves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()


plot_coming_vol_histograms(df)


"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"

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


def mean_ci(series, confidence=0.95, effective_n_divisor=1):
    """
    Confidence interval on the mean using a t-distribution,
    with an effective sample size that accounts for window overlap.
    """
    arr = series.dropna().values
    n = len(arr)
    n_eff = max(2, n // effective_n_divisor)
    if n < 2:
        m = arr.mean() if n else np.nan
        return m, m, m
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(n_eff)
    half_width = stats.t.ppf((1 + confidence) / 2, df=n_eff - 1) * sem
    return mean, mean - half_width, mean + half_width


foresight_values = list(range(1, 301, 15))
print(f"Foresight values: {foresight_values}")

results = []
for f in foresight_values:
    df_f = comp_comingvol(foresight=f, df=df_raw)

    bear_mean, bear_lo, bear_hi = mean_ci(df_f['coming_vol_br'], confidence=0.95, effective_n_divisor=f)
    bull_mean, bull_lo, bull_hi = mean_ci(df_f['coming_vol_bu'], confidence=0.95, effective_n_divisor=f)

    results.append({
        'foresight': f,
        'mean_bear': bear_mean, 'bear_lo': bear_lo, 'bear_hi': bear_hi,
        'mean_bull': bull_mean, 'bull_lo': bull_lo, 'bull_hi': bull_hi,
        'n_samples': len(df_f),
        'n_effective': max(2, len(df_f) // f),
    })
    print(f"foresight={f:4d}  bear={bear_mean:+.4f}  bull={bull_mean:+.4f}  "
          f"n={len(df_f)}  n_eff={results[-1]['n_effective']}")

results_df = pd.DataFrame(results)


def plot_means_vs_foresight(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results_df['foresight'], results_df['mean_bear'],
            color='red', label='Mean bearish (close)')
    ax.fill_between(results_df['foresight'],
                    results_df['bear_lo'], results_df['bear_hi'],
                    color='red', alpha=0.3, linewidth=0)

    ax.plot(results_df['foresight'], results_df['mean_bull'],
            color='green', label='Mean bullish (close)')
    ax.fill_between(results_df['foresight'],
                    results_df['bull_lo'], results_df['bull_hi'],
                    color='green', alpha=0.3, linewidth=0)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Foresight (number of bars)')
    ax.set_ylabel('Mean percentage move')
    ax.set_title('Average forward price move vs. foresight (95% CI, overlap-adjusted)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_means_vs_foresight(results_df)







































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

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
df_raw = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

print(df_raw)


def comp_comingvol(foresight=48, df=None, colname="coming_vol"):
    df = df.copy()  # don't mutate the input
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


def generate_foresight_values(n=20, start=1, growth=1.35):
    """
    Generate n exponentially-spaced foresight values starting at `start`.
    Removes duplicates while preserving order.
    """
    values = [int(round(start * (growth ** i))) for i in range(n)]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in values:
        if v not in seen and v >= 1:
            seen.add(v)
            unique.append(v)
    return unique


foresight_values = generate_foresight_values(n=20, start=1, growth=1.35)
print(f"Foresight values: {foresight_values}")

results = []
for f in foresight_values:
    df_f = comp_comingvol(foresight=f, df=df_raw)
    results.append({
        'foresight': f,
        'mean_bear': df_f['coming_vol_br'].mean(),
        'mean_bull': df_f['coming_vol_bu'].mean(),
        'mean_theo_bear': df_f['coming_vol_theo_br'].mean(),
        'mean_theo_bull': df_f['coming_vol_theo_bu'].mean(),
        'n_samples': len(df_f),
    })
    print(f"foresight={f:4d}  bear={results[-1]['mean_bear']:+.4f}  bull={results[-1]['mean_bull']:+.4f}  n={results[-1]['n_samples']}")

results_df = pd.DataFrame(results)
print("\n", results_df)


def plot_means_vs_foresight(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results_df['foresight'], results_df['mean_bear'],
            color='red', marker='o', label='Mean bearish (close)')
    ax.plot(results_df['foresight'], results_df['mean_bull'],
            color='green', marker='o', label='Mean bullish (close)')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Foresight (number of bars)')
    ax.set_ylabel('Mean percentage move')
    ax.set_title('Average forward price move vs. foresight window length')
    ax.set_xscale('log')  # exponential x-axis matches the exponential progression
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


plot_means_vs_foresight(results_df)