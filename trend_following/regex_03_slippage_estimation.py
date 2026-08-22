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


DB_DIR = Path("data")
DATABASE = "crypto.db"
TABLENAME = "solana1H"
PARAMETERS = [15, 25, 34, 43, 52, 61, 70, 75, 80]
OFFSET = 100
LENGHT = 10000
BIAS_LENGHT = 400
trade = None
FEE = 0.075
slippage = 0.05
MARGIN_FEE = 0.01
TOLERANCE = 0.1

db_path = DB_DIR / DATABASE
conn = sqlite3.connect(db_path)
# Load directly into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {TABLENAME}", conn)
conn.close()

# --- Ensure proper ordering (VERY IMPORTANT) ---
# Replace 'timestamp' with your actual time column if different
df = df.sort_values(by="date").reset_index(drop=True)

# --- Compute next period open ---
df["open_t_plus_1"] = df["open"].shift(-1)

# --- Drop last row (since it has no t+1) ---
df = df.dropna(subset=["open_t_plus_1"])

# --- Compute raw difference ---
df["diff_raw"] = df["close"] - df["open_t_plus_1"]

# --- Store in list ---
raw_diff_list = df["diff_raw"].tolist()

# --- Compute percentage difference relative to price at t+1 ---
df["diff_pct"] = (df["close"] - df["open_t_plus_1"]) / df["open_t_plus_1"]

# --- Store in list ---
pct_diff_list = df["diff_pct"].tolist()

def compute_range_and_mean_abs(data, percent):
    import numpy as np
    
    if not (0 < percent < 100):
        raise ValueError("percent must be between 0 and 100")
    
    data = np.array(data)
    
    # Mean
    mean = np.mean(data)
    
    # Distance from mean
    deviations = data - mean
    
    # Compute symmetric percentile bounds
    lower_pct = (100 - percent) / 2
    upper_pct = 100 - lower_pct
    
    lower_bound = np.percentile(data, lower_pct)
    upper_bound = np.percentile(data, upper_pct)
    
    # Mean absolute value
    mean_abs = np.mean(np.abs(data))
    
    print(f"Mean: {mean}")
    print(f"Mean absolute value: {mean_abs}")
    print(f"{percent}% central range: [{lower_bound}, {upper_bound}]")
    
    return lower_bound, upper_bound, mean, mean_abs

# --- Plot raw histogram ---
plt.figure()
plt.hist(raw_diff_list, bins=100, range=(-0.2, 0.2))
plt.title("Histogram of Close_t - Open_t+1 (Raw)")
plt.xlabel("Price Difference")
plt.ylabel("Frequency")
plt.show()

# --- Plot percentage histogram ---
plt.figure()
plt.hist(pct_diff_list, bins=50, range=(-0.001, 0.001))
plt.title("Histogram of Close_t - Open_t+1 (Percentage)")
plt.xlabel("Percentage Difference")
plt.ylabel("Frequency")
plt.show()