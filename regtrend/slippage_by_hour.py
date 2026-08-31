"""
Intra-candle split and range by hour of day on 5-minute SOL candles (Sept 2025).

After the second live deployment, this study measured the absolute and
signed difference between a candle's close and the next open, grouped by
hour of the day (and day of week in the journal plots), to see when the
close-to-open gap — the cost of an exit signalled at the close — is
smallest. The result motivated restricting trading hours and, more
importantly, the next-open execution model of `regtrend_backtest.py`.

Original file: hilo_028 range per hour otd.py
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DB_FILE = "data/SQL001.db"
TABLE_NAME = "Solana3"
#PARAMETER_LIST = [30, 50, 68, 86, 104, 122, 140, 150, 160]
PARAMETER_LIST = [15, 25, 34, 43, 52, 61, 70, 75, 80]
#PARAMETER_LIST = [8, 13, 17, 22, 26, 31, 35, 38, 40]
BUFFER_RATIO = 0
LENAX = 2
FEES = 0.075 # using BNB to pay fees we got 25% discount on normal 0.1% fee
PER_HOUR_FEES = 0.001
CAPITAL = 100
SL_LEVEL = 0.6
SLIP_TOLERANCE = 1 - 0.02

conn = sqlite3.connect(DB_FILE)
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
conn.close()
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.iloc[50000:-500].reset_index(drop=True)
np.set_printoptions(linewidth=120)
print(f"running on {len(df)} candles")
print(df.iloc[0])

def mean(input) :
    b = 0
    for i in range(len(input)):
        b += input[i]
    return b/len(input)
        

dt_array = df['date'].to_numpy()
high_array = df['high'].to_numpy()
low_array = df['low'].to_numpy()
close_array = df['close'].to_numpy()
open_array = df['open'].to_numpy()

hist = []
track_lst = {
    '00' : [],
    '01' : [],
    '02' : [],
    '03' : [],
    '04' : [],
    '05' : [],
    '06' : [],
    '07' : [],
    '08' : [],
    '09' : [],
    '10' : [],
    '11' : [],
    '12' : [],
    '13' : [],
    '14' : [],
    '15' : [],
    '16' : [],
    '17' : [],
    '18' : [],
    '19' : [],
    '20' : [],
    '21' : [],
    '22' : [],
    '23' : []
}
indxlst = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']


last = None
high_break, low_break, spread = 0, 0, 0
for i in range(len(df)) :
    current = dt_array[i][11:-3]
    if current != last :
        if high_break == 0 and low_break == 0 :
            pass
        else :
            spread = 100*(high_break)/low_break - 100
            track_lst[last].append(spread)
        spread, high_break, low_break = 0, high_array[i], low_array[i]
    elif current == last :
        if high_array[i] > high_break :
            high_break = high_array[i]
        if low_array[i] < low_break :
            low_break = low_array[i] 
    last = current

for i in range(len(indxlst)-1):
    plt.plot([i, i+1], [mean(track_lst[indxlst[i]]), mean(track_lst[indxlst[i + 1]])], color = 'blue')
    plt.plot([i, i+1], [(np.std(track_lst[indxlst[i]])), (np.std(track_lst[indxlst[i + 1]]))], color = 'red')
plt.grid(True)
plt.show()


fig, axes = plt.subplots(4, 6, figsize=(18, 12))
axes = axes.flatten()

for i in range(len(indxlst)):
    ax = axes[i]
    ax.hist(track_lst[indxlst[i]], bins=50, edgecolor='black')
    ax.set_title(f"Hour {indxlst[i]}")

plt.tight_layout()
plt.show()

